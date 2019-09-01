from flask import Flask, send_file
from flask_cors import CORS
from uuid import uuid4
import torch
from torch import nn
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import joblib
from torchvision import transforms
from PIL import Image

def uuid():
    return str(uuid4().hex)


# load model
ngf = 64
nc = 3
nz = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

from contextlib import contextmanager

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

class CheeHoDataset(Dataset):
    def __init__(self, thedir, tfms=None):
        self.filenames = list(map(str, list(Path(thedir).iterdir())))
        random.shuffle(self.filenames)
        self.tfms = None
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        if self.tfms is not None:
            img = self.tfms(img)
        img = transforms.Resize((224, 224))(img)
        img = transforms.ToTensor()(img)
        return img.float()


class CheeHoEncoder(nn.Module):
    def __init__(self):
        super(CheeHoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        z = F.relu(self.conv1(x))
        z = self.pool(z)
        z = F.relu(self.conv2(z))
        z = self.pool(z)
        return z
    

class CheeHoDecoder(nn.Module):
    def __init__(self):
        super(CheeHoDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        
    def forward(self, z):
        r = F.relu(self.deconv1(z))
        r = F.relu(self.deconv2(r))
        return r
    
        
class AutoCheeHoEncoder(nn.Module):
    def __init__(self):
        super(AutoCheeHoEncoder, self).__init__()
        self.encoder = CheeHoEncoder()
        self.decoder = CheeHoDecoder()
        
    def forward(self, x):
        z = self.encoder(x) # latent encoding
        r = self.decoder(z) # reconstruction
        assert x.shape == r.shape, f"input shape <{x.shape}> does not match recon shape <{r.shape}>"
        return r

netG = Generator()
netG.load_state_dict(torch.load('netGmodel.pth'))
netG.eval()

autoenc = AutoCheeHoEncoder()
autoenc.load_state_dict(torch.load('autoenc.pth'))
autoenc.eval()

pca = joblib.load('pca.joblib')
kmeans = joblib.load('kmeans.joblib')

def get_and_save_good_chee_ho() -> str:
    filename = uuid()+'.png'
    with torch.no_grad():
        while True:
            noise = torch.randn(1, nz, 1, 1)
            img = netG(noise)
            encoded = autoenc.encoder(img).reshape([1, -1]).numpy()
            pca_space = pca.transform(encoded)
            c = kmeans.predict(pca_space)
            if c == 0:
                break

        vutils.save_image(img.detach(), filename)
    return filename

# make server

app = Flask(__name__)
CORS(app)

def make_bigger(fn):
    import cv2
    img = cv2.imread(fn, 1)
    img = cv2.resize(img, (200, 200))
    cv2.imwrite(fn, img)

@app.route('/generate')
def generate():
    # noise = torch.randn(16, nz, 1, 1)
    # g = netG(noise)[0]
    # filename = uuid()+'.png'
    # vutils.save_image(g.detach(), filename)
    # # loaded = transforms.Resize(224)(Image.open(filename))
    filename = get_and_save_good_chee_ho()
    make_bigger(filename)
    return send_file(filename, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0')