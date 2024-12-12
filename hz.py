import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_to_tensor(image_path, grayscale=False):
    image = Image.open(image_path)

    what = [transforms.PILToTensor()]
    if grayscale: what.append(transforms.Grayscale())    
    transform = transforms.Compose(what)

    return transform(image).to(device)

def derivative(img):
    dx = torch.zeros(img.shape, device=device, dtype=torch.float)    
    dy = torch.zeros(img.shape, device=device, dtype=torch.float)    

    dx[:, 1:img.shape[1]-1, 1:img.shape[2]-1] = img[:, 1:img.shape[1]-1, :img.shape[2]-2] - img[:, 1:img.shape[1]-1, 2:img.shape[2]] 
    dy[:, 1:img.shape[1]-1, 1:img.shape[2]-1] = img[:, :img.shape[1]-2, 1:img.shape[2]-1] - img[:, 2:img.shape[1], 1:img.shape[2]-1] 

    return dx, dy


def zscore(im):
    s = im.std()

    if not (s==0):
        z = (im - torch.mean(im)) / s
    else:
        z[:] = 0
        
    return z


def hz(image_path, scale_base=np.sqrt(2), scale_ratio=1/np.sqrt(2), scale_th=0.75, n_scales=9, start_scale=3, dm_th=0.31, cf=3):
    img = load_to_tensor(image_path, grayscale=True).to(torch.float)

    kpt = []
    i_scale = scale_base ** np.arange(0, n_scales+1)
    d_scale = i_scale * scale_ratio
    dx, dy = derivative(img)

    for i in range(start_scale, n_scales + 1):
        rd = int(max(1, np.ceil( 3 * d_scale[i])))
        hd = 2 * rd + 1

        dx_d = v2.Pad(padding=rd, padding_mode='reflect')(dx)
        dy_d = v2.Pad(padding=rd, padding_mode='reflect')(dy)
        
        dx_d = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])(dx_d)[:, rd:-rd, rd:-rd]
        dy_d = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])(dy_d)[:, rd:-rd, rd:-rd]

        dm = (dx_d**2 + dy_d**2)**0.5;

        dm_mask = (dm > dm.mean()).to(torch.float)
        dm_mask = v2.Pad(padding=rd, padding_mode='reflect')(dm_mask)
        dm_mask = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])(dm_mask)[:, rd:-rd, rd:-rd]

        dx_d = dx_d * dm_mask
        dy_d = dy_d * dm_mask
        
        ri = int(max(1, np.ceil(3 * i_scale[i])))
        hi = 2 * ri + 1

        dxy = v2.Pad(padding=ri, padding_mode='reflect')(dx_d * dy_d)
        dxy = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])(dxy)[:, ri:-ri, ri:-ri]

        dx2 = v2.Pad(padding=ri, padding_mode='reflect')(dx_d * dx_d)
        dx2 = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])(dx2)[:, ri:-ri, ri:-ri]

        dy2 = v2.Pad(padding=ri, padding_mode='reflect')(dy_d * dy_d)
        dy2 = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])(dy2)[:, ri:-ri, ri:-ri]

        harris = zscore(dx2 * dy2 - dxy**2) - zscore((dx2 + dy2)**2)
    return

if __name__ == '__main__':
    pts = hz('images/graf5.png')