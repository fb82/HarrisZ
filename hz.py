import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

import cv2
import kornia.feature as KF
import kornia as K
from kornia_moons.viz import visualize_LAF
import matplotlib.pyplot as plt

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

def max_mask(img, rd, dm_mask, rad_max=3):
    rad = min(rad_max, max(1, round(rd / np.sqrt(2))));
    k = 2 * rad + 1

    max_img = torch.nn.MaxPool2d(k, stride=1, padding=rad)(img) == img
    unique_max_img = ((torch.nn.AvgPool2d(k, stride=1, padding=rad)(max_img.type(torch.float)) * k**2) == 1) & max_img
    ismax = unique_max_img & (img > 0) & dm_mask

    rc = torch.argwhere(ismax)
    if rc.shape[0] == 0:
        return torch.zeros((0,2), device=device, dtype=torch.int)
    
    rc_flat_index = rc[:, 1] * img.shape[2] + rc[:, 2]
    max_val = img.flatten()[rc_flat_index]
    sidx = torch.argsort(max_val, descending=True)
    rc = rc[sidx, 1:]    

    m = torch.cdist(rc.type(torch.float), rc.type(torch.float))
    m_idx = torch.full((rc.shape[0],), -1, device=device, dtype=torch.int)
    m_idx[0] = 0
    m_n = 1
    for i in range(1,m_idx.shape[0]):
        if (m[i, m_idx[:m_n]] >= rd).all():
            m_n = m_n + 1
            m_idx[i] = i

    return rc[m_idx[:m_n]]
        

def sub_pix(img, kp):
    kp_sp = kp.type(torch.float)
    
    r_mask = (kp[:,0] > 1) & (kp[:, 0] < img.shape[0] - 1)

    v  = img.flatten()[ kp[r_mask, 0]      * img.shape[2] + kp[r_mask, 1]]
    vl = img.flatten()[(kp[r_mask, 0] - 1) * img.shape[2] + kp[r_mask, 1]]
    vr = img.flatten()[(kp[r_mask, 0] + 1) * img.shape[2] + kp[r_mask, 1]]

    kp_sp[r_mask, 0] = kp[r_mask, 0] + (vr - vl) / (2 * (2*v - vl -vr))    

    c_mask = (kp[:,1] > 1) & (kp[:, 1] < img.shape[1] - 1)

    v  = img.flatten()[kp[c_mask, 0] * img.shape[2] + kp[c_mask, 1]    ]
    vl = img.flatten()[kp[c_mask, 0] * img.shape[2] + kp[c_mask, 1] - 1]
    vr = img.flatten()[kp[c_mask, 0] * img.shape[2] + kp[c_mask, 1] + 1]

    kp_sp[c_mask, 1] = kp[c_mask, 1] + (vr - vl) / (2 * (2*v - vl -vr))    

    return kp_sp
    

def get_eli(dx2, dy2, dxy, scale):
    U = torch.zeros((dx2.shape[0], 2, 2), device=device)
    U[:, 0, 0] = dx2
    U[:, 0, 1] = dxy
    U[:, 1, 0] = dxy
    U[:, 1, 1] = dy2
    D, V = torch.linalg.eigh(U)
    D = 1 / D**0.5
    D = D / D[:, 0].unsqueeze(-1)    
    kp_ratio = torch.minimum(D[:, 0], D[:, 1])
    D = D * scale
    D_ = torch.zeros((dx2.shape[0], 2, 2), device=device)
    D_[:, 0, 0] = 1 / D[:, 0]**2
    D_[:, 1, 1] = 1 / D[:, 1]**2
    U_ = V @ D_ @ V.transpose(1, 2) 
    kp_eli = torch.stack((U_[:, 0, 0], U_[:, 0, 1], U_[:, 1, 1]), dim=1)    

    return kp_eli, kp_ratio


def hz(img, scale_base=np.sqrt(2), scale_ratio=1/np.sqrt(2), scale_th=0.75, n_scales=9, start_scale=3, dm_th=0.31, cf=3, output_format='vgg'):
    kpt = torch.zeros((0,9), device=device)
    i_scale = scale_base ** np.arange(0, n_scales+1)
    d_scale = i_scale * scale_ratio
    dx, dy = derivative(img)

    for i in range(start_scale, n_scales + 1):
        rd = int(max(1, np.ceil(3 * d_scale[i])))
        hd = 2 * rd + 1
        
        dx_d = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])(dx)
        dy_d = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])(dy)

        dm = (dx_d**2 + dy_d**2)**0.5;
        dm_mask = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])((dm > dm.mean()).to(torch.float))

        dx_d = dx_d * dm_mask
        dy_d = dy_d * dm_mask
        
        ri = int(max(1, np.ceil(3 * i_scale[i])))
        hi = 2 * ri + 1

        dxy = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])(dx_d * dy_d)
        dx2 = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])(dx_d**2)
        dy2 = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])(dy_d**2)

        harris = zscore(dx2 * dy2 - dxy**2) - zscore((dx2 + dy2)**2)
        
        kp = max_mask(harris, rd, dm_mask > dm_th)
        
        if kp.shape[0] == 0:
            continue
        
        kp_sub_pix = sub_pix(harris, kp)
        kp_s = torch.tensor([i, d_scale[i], i_scale[i]], device=device).repeat(kp.shape[0], 1)
        
        kp_index = kp[:, 0] * harris.shape[2] + kp[:, 1]
        kp_eli, kp_ratio = get_eli(dx2.flatten()[kp_index],dy2.flatten()[kp_index],dxy.flatten()[kp_index], i_scale[i] * cf)
        
        kpt_ = torch.cat((kp_sub_pix[:,[1, 0]], kp_eli, kp_s, kp_ratio.unsqueeze(-1)), dim=1)
        kp_good = 1 - kp_ratio < scale_th
 
        kpt = torch.cat((kpt, kpt_[kp_good]))

    if output_format == 'vgg':
        aux = torch.linalg.inv(kpt[:, [2, 3, 3, 4]].reshape(-1, 2, 2))
        kpt[:, 2:5] = aux.reshape(-1, 4)[:, [0, 1, 3]]
    else:
        kpt[:, 2:5] = kpt[:, 2:5] / cf

    return kpt[:, :5]

if __name__ == '__main__':
    image = 'images/graf5.png'

    img = load_to_tensor(image, grayscale=True).to(torch.float)
    pts = hz(img, output_format='laf')

    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    lafs = KF.ellipse_to_laf(pts[None])     
    visualize_LAF(K.image_to_tensor(img, False), lafs, 0, return_fig_ax=True)    
    plt.savefig('harrisz_pytorch.pdf', dpi = 150, bbox_inches='tight')