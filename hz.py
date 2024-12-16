# for HarrisZ
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

# for demo
import sys

# for visualization
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# for Kornia
try:
    import kornia as K
    import kornia.feature as KF
    from kornia_moons.viz import visualize_LAF
    import cv2
    kornia_on = True
except:
    kornia_on = False
    import warnings
    warnings.warn("Kornia e Kornia-Moons not found: skipping the related demo part")    
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_hz_eli(image, kpts, save_to='harrisz_pytorch_eli.pdf', dpi=150, c_color='b', c_marker='.', markersize=1, e_color='b', linewidth=0.5):
    plt.figure()
    plt.axis('off')
    plt.imshow(Image.open(image))

    pts = kpts['center'].to('cpu').numpy()
    eli_axes =kpts['axes'].to('cpu').numpy()
    eli_rot = kpts['rotation'].to('cpu').numpy()
    
    ax = plt.gca()
    plt.plot(pts[:, 0], pts[:, 1], linestyle='', color=c_color, marker=c_marker, markersize=markersize)
    for i in range(pts.shape[0]):
        eli = Ellipse(xy=(pts[i, 0], pts[i, 1]), width=eli_axes[i, 0], height=eli_axes[i, 1], angle=eli_rot[i], facecolor='none', edgecolor=e_color, linewidth=linewidth)
        ax.add_patch(eli)

    if not(save_to is None):
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')


def load_to_tensor(image_path, grayscale=False):
    image = Image.open(image_path)

    what = [transforms.PILToTensor()]
    if grayscale: what.append(transforms.Grayscale())    
    transform = transforms.Compose(what)

    return transform(image).to(device)

def derivative(img):
    dx = torch.nn.functional.pad(img[:, 1:img.shape[1]-1, :img.shape[2]-2] - img[:, 1:img.shape[1]-1, 2:img.shape[2]],(1,1,1,1,0,0), mode='constant', value=0) 
    dy = torch.nn.functional.pad(img[:, :img.shape[1]-2, 1:img.shape[2]-1] - img[:, 2:img.shape[1], 1:img.shape[2]-1],(1,1,1,1,0,0), mode='constant', value=0) 

    return dx, dy


def zscore(im):
    s = im.std()

    if not (s==0):
        z = (im - torch.mean(im)) / s
    else:
        z = torch.zeros_like(im)
        
    return z

def max_mask(img, rd, dm_mask, rad_max=3, block_mem=16*10**6):
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

    m_idx = torch.full((rc.shape[0],), -1, device=device, dtype=torch.int)
    m_idx[0] = 0
    m_n = 1

#   # base
#   m = torch.cdist(rc.type(torch.float), rc.type(torch.float))
#   for i in range(1,m_idx.shape[0]):
#       if (m[i, m_idx[:m_n]] >= rd).all():
#           m_n = m_n + 1
#           m_idx[i] = i

    # memory-optimized
    # note that when m_n_ == m_n -->
    # (m[i - ii, m_idx[m_n_:m_n]] >= rd).all() = ([]).all() = True
    ii = 1
    while ii < m_idx.shape[0]:
        block_len = np.ceil((np.sqrt(ii**2 + 4 * block_mem) - ii) * 0.5)
        ij = int(min(m_idx.shape[0], ii + block_len))
        m = torch.cdist(rc[ii:ij].type(torch.float), rc.type(torch.float))
        to_check = (m[:, m_idx[:m_n]] >= rd).all(dim=1)
        m_n_ = m_n
        for i in range(ii, ij):
            if (to_check[i - ii]) and (m[i - ii, m_idx[m_n_:m_n]] >= rd).all():
                m_n = m_n + 1
                m_idx[i] = i
        ii = ij

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
    U = torch.stack((dx2, dxy, dxy, dy2), dim=-1).reshape(-1, 2, 2)    
    D, V = torch.linalg.eigh(U)
    D = 1 / D**0.5
    D = D / D[:, 0].unsqueeze(-1)
    kp_ratio = torch.minimum(D[:, 0], D[:, 1])
    D = torch.diag_embed(1 / (D * scale)**2)
    U = V @ D @ V.transpose(1, 2) 
    kp_eli = torch.stack((U[:, 0, 0], U[:, 0, 1], U[:, 1, 1]), dim=1)    

    return kp_eli, kp_ratio


def hz(img, scale_base=np.sqrt(2), scale_ratio=1/np.sqrt(2), scale_th=0.75, n_scales=9, start_scale=3, dm_th=0.31, cf=3, xy_offset=0.0, output_format='vgg'):
    kpt = torch.zeros((0,9), device=device)
    i_scale = scale_base ** np.arange(0, n_scales)
    d_scale = i_scale * scale_ratio
    dx, dy = derivative(img)

    for i in range(start_scale, n_scales):        
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

    kpt[:, :2] = kpt[:, :2] + xy_offset
    if output_format == 'vgg':
        kpt[:, 2:5] = torch.linalg.inv(kpt[:, [2, 3, 3, 4]].reshape(-1, 2, 2)).reshape(-1, 4)[:, [0, 1, 3]]
        return kpt[:, :5]
    elif output_format == 'laf':
        kpt[:, 2:5] = kpt[:, 2:5] / cf
        return kpt[:, :5]        
    else:
        D, V = torch.linalg.eigh(kpt[:, [2, 3, 3, 4]].reshape(-1, 2, 2))
        center = kpt[:, :2]
        axes = (cf / D)**0.5
        rotation = torch.atan2(V[:, 1, 0], V[:, 0, 0]) * 180 / np.pi
        return {'center': center, 'axes': axes, 'rotation': rotation}


def hz_plus(img, max_kpts=8000, fast_save_memory=False, scale_base=np.sqrt(2), scale_ratio=1/np.sqrt(2), scale_th=0.75,
       n_scales=4, start_scale=0, dm_th=0.31, cf=3, xy_offset=0.0, output_format='vgg',
       start_scale_at_2x=2, rescale_method=Image.Resampling.LANCZOS, min_scale=np.sqrt(2), sieve_rad=1, laf_offset=10, max_kpts_cf=2):
    
    sz = img.shape
    if sz[0] == 3: color_grad=True
    else: color_grad=False
    
    if start_scale < start_scale_at_2x:
        im_2x = transforms.PILToTensor()(transforms.ToPILImage()(img).resize((sz[2] * 2, sz[1] * 2), resample=rescale_method)).to(torch.float)

    kpt = torch.zeros((0,9), device=device)
    if color_grad:        
        img1 = transforms.functional.rgb_to_grayscale(img)
        img2 = torch.max(img, dim=0)[0][None]
        dx1, dy1 = derivative(img1);
        dx2, dy2 = derivative(img2);

    return
    
if __name__ == '__main__':
    # example image
    image = 'images/graf5.png'

    # a diffent image can be passed to the demo script
    if len(sys.argv) > 1:
        image = sys.argv[1]

    ### HarrisZ+

    # standalone usage
    img = load_to_tensor(image).to(torch.float)
    start = time.time()
    kpts = hz_plus(img, output_format='eli')    
    end = time.time()
    print("Elapsed = %s (HarrisZ)" % (end - start))
    # show keypoints 
    plot_hz_eli(image, kpts, save_to='harrisz_pytorch_eli.pdf')

    ### HarrisZ

    # standalone usage
    img = load_to_tensor(image, grayscale=True).to(torch.float)
    start = time.time()
    kpts = hz(img, output_format='eli')    
    end = time.time()
    print("Elapsed = %s (HarrisZ)" % (end - start))
    # show keypoints 
    plot_hz_eli(image, kpts, save_to='harrisz_pytorch_eli.pdf')
    
    # with Kornia
    if kornia_on:
        # run and convert to laf
        img = load_to_tensor(image, grayscale=True).to(torch.float)
        kpts = hz(img, output_format='laf')
        lafs = KF.ellipse_to_laf(kpts[None]) 

        # show keypoints with Kornia
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)    
        visualize_LAF(K.image_to_tensor(img, False), lafs, 0)
        plt.axis('off')    

        # save the plot        
        plt.savefig('harrisz_pytorch_laf.pdf', dpi=150, bbox_inches='tight')
