# for HarrisZ
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tt
from torchvision.transforms import v2

# for demo
import sys
import os

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
        eli = Ellipse(xy=(pts[i, 0], pts[i, 1]), width=eli_axes[i, 0], height=eli_axes[i, 1], angle=eli_rot[i],
                      facecolor='none', edgecolor=e_color, linewidth=linewidth)
        ax.add_patch(eli)

    if not(save_to is None):
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')


def load_to_tensor(image_path, grayscale=False):
    image = Image.open(image_path)

    what = [tt.PILToTensor()]
    if grayscale: what.append(tt.Grayscale())    
    transform = tt.Compose(what)

    return transform(image).to(device)

def derivative(img):
    dx_dy = torch.cat((img[:, 1:img.shape[1]-1, :img.shape[2]-2] - img[:, 1:img.shape[1]-1, 2:img.shape[2]],
                 img[:, :img.shape[1]-2, 1:img.shape[2]-1] - img[:, 2:img.shape[1], 1:img.shape[2]-1]), dim=0)
    dx_dy = torch.nn.functional.pad(dx_dy,(1,1,1,1,0,0), mode='constant', value=0) 
    return dx_dy


def zscore(im):
    s = im.std()

    if not (s==0):
        z = (im - torch.mean(im)) / s
    else:
        z = torch.zeros_like(im)
        
    return z

def max_mask(img, rd, dm_mask, rad_max=3, block_mem=16*10**6, prev_filter={'k': 0}, max_max_pts=np.inf):
    rad = min(rad_max, max(1, round(rd / np.sqrt(2))));
    k = 2 * rad + 1

    if prev_filter['k'] == k:
        max_filter = prev_filter['max_filter']
        avg_filter = prev_filter['avg_filter']
    else:    
        max_filter = torch.nn.MaxPool2d(k, stride=1, padding=rad)
        avg_filter = torch.nn.AvgPool2d(k, stride=1, padding=rad)
    
    max_img = max_filter(img) == img
    unique_max_img = ((avg_filter(max_img.type(torch.float)) * k**2) == 1) & max_img
    ismax = unique_max_img & (img > 0) & dm_mask

    rc = torch.argwhere(ismax)
    if rc.shape[0] == 0:
        return torch.zeros((0,2), device=device, dtype=torch.int)
    
    rc_flat_index = rc[:, 1] * img.shape[2] + rc[:, 2]
    max_val = img.flatten()[rc_flat_index]
    sidx = torch.argsort(max_val, descending=True)
    rc = rc[sidx, 1:]    

    rc_idx = select_max(rc, rd, block_mem=block_mem, max_max_pts=max_max_pts)
    return rc[rc_idx], {'k': k, 'max_filter': max_filter, 'avg_filter': avg_filter}
        

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


def best_derivative(dx_dy_1, dx_dy_2):
    aux = aux =torch.cat((dx_dy_1, dx_dy_2), dim=1).reshape(2,2,dx_dy_1.shape[-2],dx_dy_1.shape[-1])
    _, aux_abs = aux.abs().max(dim=1)
    return aux.gather(dim=1, index=aux_abs[:, None]).squeeze(1)


# base version v0 - subject to OOM
def select_max_v0(rc, rd, block_mem=16*10**6, max_max_pts=np.inf):    
    m_idx = torch.zeros(rc.shape[0], device=device, dtype=torch.int)
    m_idx[0] = 0
    m_n = 1

    m = torch.cdist(rc.type(torch.float), rc.type(torch.float))
    for i in range(1,m_idx.shape[0]):
        if (m[i, m_idx[:m_n]] >= rd).all():
            m_n = m_n + 1
            m_idx[i] = i
            if m_n > max_max_pts: break

    return m_idx[:m_n]     


# memory-optimized v1 - no OOM issues
# note that when m_n_ == m_n -->
# (m[i - ii, m_idx[m_n_:m_n]] >= rd).all() = ([]).all() = True
def select_max_v1(rc, rd, block_mem=16*10**6, max_max_pts=np.inf):    
    m_idx = torch.zeros(rc.shape[0], device=device, dtype=torch.int)
    m_idx[0] = 0
    m_n = 1

    ii = 1
    while ii < m_idx.shape[0]:
        block_len = np.ceil((np.sqrt(ii**2 + 4 * block_mem) - ii) * 0.5)
        ij = int(min(m_idx.shape[0], ii + block_len))
        m = torch.cdist(rc[ii:ij].type(torch.float), rc.type(torch.float))
        to_check = (m[:, m_idx[:m_n]] >= rd).all(dim=1)
        m_n_ = m_n
        for i in range(ii, ij):
            if (to_check[i - ii]) and (m[i - ii, m_idx[m_n_:m_n]] >= rd).all():
                m_idx[m_n] = i
                m_n = m_n + 1
                if m_n > max_max_pts: break
                
        ii = ij

    return m_idx[:m_n]     


# memory-optimized v2 - no OOM, actually worse than v1
# note that when m_n_ == m_n -->
# (m[i - ii, m_idx[m_n_:m_n] - ii] >= rd).all() = (m[i - ii, [] - ii] >= rd).all() = ([]).all() = True    
def select_max_v2(rc, rd, block_mem=16*10**6, max_max_pts=np.inf):    
    m_idx = torch.zeros(rc.shape[0], device=device, dtype=torch.int)
    m_idx[0] = 0
    m_n = 1

    block_len = int(np.ceil(np.sqrt(block_mem)))
    ii = 1
    for ii in np.arange(1, m_idx.shape[0], block_len):
        check_bad = torch.zeros(min(m_idx.shape[0], ii + block_len) - ii, device=device, dtype=torch.bool)        
        
        ij = int(min(m_idx.shape[0], ii + block_len))
        for i in np.arange(0, ii, block_len):
            j = int(min(ii, i + block_len))
            m = torch.cdist(rc[ii:ij].type(torch.float), rc[i:j].type(torch.float))
            check_bad = check_bad | (m < rd).all(dim=1)
       
        m = torch.cdist(rc[ii:ij].type(torch.float), rc[ii:ij].type(torch.float))
        m_n_ = m_n
        for i in np.arange(ii, ij):
            if (not (check_bad[i - ii])) and (m[i - ii, m_idx[m_n_:m_n] - ii] >= rd).all():
                m_idx[m_n] = i
                m_n = m_n + 1
                if m_n > max_max_pts: break

    return m_idx[:m_n]     

select_max = select_max_v1

def uniform_kpts(sz, kpt, max_kpts, max_kpts_cf, max_max_pts=np.inf, block_mem=16*10**6):
    c_kp = torch.zeros((0, 10), device=device)
    r_kp = kpt
    c_d = 2 * np.sqrt(sz[1] * sz[2] / (max_kpts * np.pi/ max_kpts_cf))
    
    while True:
        if r_kp.shape[0] == 0:
            break

        idx = r_kp[:, 5].argsort(descending=False)
        idx_ = r_kp[idx, 9].argsort(descending=False, stable=True)
        idx = idx[idx_]
        
        r_kp = r_kp[idx]
        max_index = select_max(r_kp[:, :2], c_d, block_mem=block_mem, max_max_pts=max_max_pts)
        c_idx = torch.zeros(r_kp.shape[0], device=device, dtype=torch.bool)
        c_idx[max_index] = True

        c_kp = torch.cat((c_kp, r_kp[c_idx]))
        r_kp = r_kp[~c_idx]
        
        if c_kp.shape[0] > max_max_pts:
            c_kp = c_kp[:max_max_pts]
            break

    return c_kp        


def hz(img, scale_base=np.sqrt(2), scale_ratio=1/np.sqrt(2), scale_th=0.75, n_scales=9, start_scale=3,
       dm_th=0.31, cf=3, xy_offset=0.0, max_max_pts=np.inf, block_mem=16*10**6, output_format='vgg'):
    kpt = torch.zeros((0, 10), device=device)
    i_scale = scale_base ** np.arange(0, n_scales)
    d_scale = i_scale * scale_ratio
    dx_dy = derivative(img)

    prev_filter={'k': 0}
    for i in range(start_scale, n_scales):        
        rd = int(max(1, np.ceil(3 * d_scale[i])))
        hd = 2 * rd + 1
        
        smooth_d = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])
        dx_dy_d = smooth_d(dx_dy)

        dm = ((dx_dy_d**2).sum(dim=0, keepdim=True)**0.5)
        dm_mask = smooth_d((dm > dm.mean()).to(torch.float))

        dx_dy_d = dx_dy_d * dm_mask
        
        ri = int(max(1, np.ceil(3 * i_scale[i])))
        hi = 2 * ri + 1

        smooth_i = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])
        dxy = smooth_i(dx_dy_d.prod(dim=0, keepdim=True))
        dx2_dy2 = smooth_i(dx_dy_d**2)

        harris = zscore(dx2_dy2.prod(dim=0, keepdim=True) - dxy**2) - zscore((dx2_dy2.sum(dim=0, keepdim=True))**2)
        
        kp, prev_filter = max_mask(harris, rd, dm_mask > dm_th, prev_filter=prev_filter, max_max_pts=max_max_pts, block_mem=block_mem)
        
        if kp.shape[0] == 0:
            continue
        
        kp_sub_pix = sub_pix(harris, kp)
        kp_s = torch.tensor([i, d_scale[i], i_scale[i]], device=device).repeat(kp.shape[0], 1)
        
        kp_index = kp[:, 0] * harris.shape[2] + kp[:, 1]
        kp_eli, kp_ratio = get_eli(dx2_dy2[0].flatten()[kp_index], dx2_dy2[1].flatten()[kp_index], dxy.flatten()[kp_index], i_scale[i] * cf)
        hv = harris.flatten()[kp_index]        
        
        kpt_ = torch.cat((kp_sub_pix[:,[1, 0]], kp_eli, kp_s, kp_ratio.unsqueeze(-1), hv.unsqueeze(-1)), dim=1)
        kp_good = 1 - kp_ratio < scale_th
 
        kpt = torch.cat((kpt, kpt_[kp_good]))

    idx = kpt[:, 5].argsort(descending=False, stable=True)
    kpt = kpt[idx]
    if kpt.shape[0] > max_max_pts:    
        kpt = kpt[:max_max_pts]

    kpt[:, :2] = kpt[:, :2] + xy_offset
    if output_format == 'vgg':
        kpt[:, 2:5] = torch.linalg.inv(kpt[:, [2, 3, 3, 4]].reshape(-1, 2, 2)).reshape(-1, 4)[:, [0, 1, 3]]
        return kpt[:, :5], kpt[:, -1]
    elif output_format == 'laf':
        kpt[:, 2:5] = kpt[:, 2:5] / cf
        return kpt[:, :5], kpt[:, -1]        
    else:
        D, V = torch.linalg.eigh(kpt[:, [2, 3, 3, 4]].reshape(-1, 2, 2))
        center = kpt[:, :2]
        axes = (cf / D)**0.5
        rotation = torch.atan2(V[:, 1, 0], V[:, 0, 0]) * 180 / np.pi
        return {'center': center, 'axes': axes, 'rotation': rotation, 'response': kpt[:, -1]}


def hz_plus(img, max_kpts=8000, fast_save_memory=False, scale_base=np.sqrt(2), scale_ratio=1/np.sqrt(2), scale_th=0.75,
       n_scales=4, start_scale=0, dm_th=0.31, cf=3, xy_offset=0.0, output_format='vgg', max_max_pts=np.inf, block_mem=16*10**6, color_grad=True,
       start_scale_at_2x=2, rescale_method=Image.Resampling.LANCZOS, min_scale=np.sqrt(2), sieve_rad=1, laf_offset=10, max_kpts_cf=2):
    
    sz = img.shape
    if sz[0] != 3: color_grad=False
    
    if start_scale < start_scale_at_2x:
        img_2x = tt.PILToTensor()(tt.ToPILImage()(img).resize((sz[2] * 2, sz[1] * 2), resample=rescale_method)).to(torch.float).to(device)

    kpt = torch.zeros((0,9), device=device)
    if color_grad:        
        img1 = tt.functional.rgb_to_grayscale(img)
        img2 = torch.amax(img, dim=0, keepdim=True)
        dx_dy_1 = derivative(img1);
        dx_dy_2 = derivative(img2);
        dx_dy_g = best_derivative(dx_dy_1, dx_dy_2)
        dx_dy = dx_dy_1
        
        if start_scale < start_scale_at_2x:
            img1_2x = tt.functional.rgb_to_grayscale(img_2x)
            img2_2x = torch.amax(img_2x, dim=0, keepdim=True)
            dx_dy_1_2x = derivative(img1_2x);
            dx_dy_2_2x = derivative(img2_2x);
            dx_dy_g_2x = best_derivative(dx_dy_1_2x, dx_dy_2_2x)
            dx_dy_2x = dx_dy_1_2x
    else:                
        if sz[0] == 3:
            img = tt.functional.rgb_to_grayscale(img)
        dx_dy = derivative(img)     
    
        if start_scale < start_scale_at_2x:
            if sz[0] == 3:
                img_2x = tt.functional.rgb_to_grayscale(img_2x)
            dx_dy_2x = derivative(img_2x);     
        
    i_scale = scale_base**np.arange(start_scale, n_scales+1)
    d_scale = i_scale * scale_ratio;
    is_2x_scale = np.arange(start_scale, n_scales+1) < start_scale_at_2x;       
        
    dx_dy_1x = dx_dy
    if color_grad:
        dx_dy_g_1x = dx_dy_g
        
    kpt = torch.zeros((0, 10), device=device)        
    prev_filter={'k': 0} 
    for i in range(len(i_scale)):
        if not is_2x_scale[i]:
            dx_dy = dx_dy_1x
            if color_grad: dx_dy_g = dx_dy_g_1x
        else:
            dx_dy = dx_dy_2x
            if color_grad: dx_dy_g = dx_dy_g_2x

            d_scale[i] = d_scale[i] * 2;
            i_scale[i] = i_scale[i] * 2;

        rd = int(max(1, np.ceil(3 * d_scale[i])))
        hd = 2 * rd + 1

        smooth_d = v2.GaussianBlur(kernel_size=hd, sigma=d_scale[i])
        dx_dy_d = smooth_d(dx_dy)
        if not color_grad:
            dm = ((dx_dy_d**2).sum(dim=0, keepdim=True)**0.5)
            dm_mask = smooth_d((dm > dm.mean()).to(torch.float))
        else:
            dx_dy_g_d = smooth_d(dx_dy_g)
            dm = ((dx_dy_g_d**2).sum(dim=0, keepdim=True)**0.5)
            dm_mask = smooth_d((dm > dm.mean()).to(torch.float))

        dx_dy_d = dx_dy_d * dm_mask
        ri = int(max(1, np.ceil(3 * i_scale[i])))
        hi = 2 * ri + 1

        smooth_i = v2.GaussianBlur(kernel_size=hi, sigma=i_scale[i])
        dxy = smooth_i(dx_dy_d.prod(dim=0, keepdim=True))
        dx2_dy2 = smooth_i(dx_dy_d**2)

        harris = zscore(dx2_dy2.prod(dim=0, keepdim=True) - dxy**2) - zscore((dx2_dy2.sum(dim=0, keepdim=True))**2)
        kp, prev_filter = max_mask(harris, rd, dm_mask > dm_th, prev_filter=prev_filter, max_max_pts=max_max_pts, block_mem=block_mem)
        
        if kp.shape[0] == 0:
            continue

        if not is_2x_scale[i]:
            double_adjust = 1
        else:
            double_adjust = 2
        d_scale_ = max(min_scale * scale_ratio, d_scale[i] / double_adjust)
        i_scale_ = max(min_scale, i_scale[i] / double_adjust)            
        
        kp_sub_pix = sub_pix(harris, kp)
        kp_s = torch.tensor([i, d_scale_, i_scale_], device=device).repeat(kp.shape[0], 1)

        kp_index = kp[:, 0] * harris.shape[2] + kp[:, 1]
        kp_eli, kp_ratio = get_eli(dx2_dy2[0].flatten()[kp_index], dx2_dy2[1].flatten()[kp_index], dxy.flatten()[kp_index], i_scale_ * cf)
        hv = harris.flatten()[kp_index]
        
        kpt_ = torch.cat((kp_sub_pix[:,[1, 0]] / double_adjust, kp_eli, kp_s, kp_ratio.unsqueeze(-1), hv.unsqueeze(-1)), dim=1)
        kp_good = 1 - kp_ratio < scale_th 

        kpt = torch.cat((kpt, kpt_[kp_good]))
        
    if min_scale:
        kpt = kpt[torch.argsort(kpt[:, -1], descending=True)]
        to_check = kpt[:, 6] == min_scale * scale_ratio
        to_hold_idx = select_max(kpt[to_check, :2], sieve_rad, block_mem=block_mem, max_max_pts=max_max_pts)
        to_remove = torch.full((to_check.sum(), ), 1, device=device, dtype=torch.bool)
        to_remove[to_hold_idx] = False
        to_check[to_check.clone()] = to_remove
        kpt = kpt[~to_check]

    kpt = uniform_kpts(sz, kpt, max_kpts, max_kpts_cf, max_max_pts=max_max_pts, block_mem=block_mem)

    kpt[:, :2] = kpt[:, :2] + xy_offset
    if output_format == 'vgg':
        kpt[:, 2:5] = torch.linalg.inv(kpt[:, [2, 3, 3, 4]].reshape(-1, 2, 2)).reshape(-1, 4)[:, [0, 1, 3]]
        return kpt[:, :5], kpt[:, -1]
    elif output_format == 'laf':
        kpt[:, 2:5] = kpt[:, 2:5] / cf
        return kpt[:, :5], kpt[:, -1]        
    else:
        D, V = torch.linalg.eigh(kpt[:, [2, 3, 3, 4]].reshape(-1, 2, 2))
        center = kpt[:, :2]
        axes = (cf / D)**0.5
        rotation = torch.atan2(V[:, 1, 0], V[:, 0, 0]) * 180 / np.pi
        return {'center': center, 'axes': axes, 'rotation': rotation, 'response': kpt[:, -1]}

    
if __name__ == '__main__':
    # example image
    image = 'images/graf5.png'
#   image = 'images/s_peter.png'
#   image = 'images/wooden_lady.jpg'

    # a diffent image can be passed to the demo script
    if len(sys.argv) > 1:
        image = sys.argv[1]
        
    iname, iext = os.path.splitext(image)        
    
    block_memory = 16*10**6 
    max_pts = 8000 # np.inf 

    print(f"Image: {image} (other images can be passed as 1st argument of the script)")
    print(f"Memory block dimension: {block_memory} floats (reduce in case of OOM)")
    print(f"Max number of keypoints to extract: {max_pts} (reduce for faster computation, especially with bigger images)")
    print("Note: 1. returned keypoints are sorted from the best to the worst")
    print("      2. by default all keypoints are returned, setting the related parameter to inf")
    print("      3. HarrisZ input image is only grayscale, HarrisZ+ works usually better with RGB images")
    print("")

    ### HarrisZ
    print("Running HarrisZ standalone")
    # standalone usage
    img = load_to_tensor(image, grayscale=True).to(torch.float)
    start = time.time()
    kpts = hz(img, output_format='eli', block_mem=block_memory, max_max_pts=max_pts)    
    end = time.time()
    print(f"Extracted keypoints: {kpts['center'].shape[0]}")
    print("Elapsed time: %s (HarrisZ)" % (end - start))
    # show keypoints 
    to_save = iname + '_harrisz.pdf'    
    print(f"Plot keypoint ellipses and save the result in {to_save}")
    start = time.time()
    plot_hz_eli(image, kpts, save_to=to_save)
    end = time.time()
    print("Elapsed time: %s (plot)" % (end - start))
    print("")

    # with Kornia
    if kornia_on:
        print("Running HarrisZ and exporting to Kornia format")
        # run and convert to laf
        img = load_to_tensor(image, grayscale=True).to(torch.float)
        start = time.time()
        kpts, responses = hz(img, output_format='laf', block_mem=block_memory, max_max_pts=max_pts)
        lafs = KF.ellipse_to_laf(kpts[None]) 
        end = time.time()
        print(f"Extracted keypoints: {kpts.shape[0]}")
        print("Elapsed time: %s (HarrisZ)" % (end - start))
        # show keypoints 
        to_save = iname + '_harrisz_kornia.pdf'    
        print(f"Plot keypoint ellipses in Kornia and save the result in {to_save}")
        start = time.time()
        # show keypoints with Kornia
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)    
        visualize_LAF(K.image_to_tensor(img, False), lafs, 0)
        plt.axis('off')    
        # save the plot        
        plt.savefig(to_save, dpi=150, bbox_inches='tight')
        end = time.time()
        print("Elapsed time: %s (plot)" % (end - start))
        print("")
    
    ### HarrisZ+
    print("Running HarrisZ+ standalone")
    # standalone usage
    img = load_to_tensor(image).to(torch.float)
    start = time.time()
    kpts = hz_plus(img, output_format='eli', block_mem=block_memory, max_max_pts=max_pts)    
    end = time.time()
    print(f"Extracted keypoints: {kpts['center'].shape[0]}")
    print("Elapsed time: %s (HarrisZ+)" % (end - start))
    # show keypoints 
    to_save = iname + '_harrisz_plus.pdf'    
    print(f"Plot keypoint ellipses and save the result in {to_save}")
    start = time.time()
    plot_hz_eli(image, kpts, save_to=to_save)
    end = time.time()
    print("Elapsed time: %s (plot)" % (end - start))
    print("")

    # with Kornia
    if kornia_on:
        print("Running HarrisZ+ and exporting to Kornia format")
        # run and convert to laf
        img = load_to_tensor(image).to(torch.float)
        start = time.time()
        kpts, responses = hz_plus(img, output_format='laf', block_mem=block_memory, max_max_pts=max_pts)
        lafs = KF.ellipse_to_laf(kpts[None]) 
        end = time.time()
        print(f"Extracted keypoints: {kpts.shape[0]}")
        print("Elapsed time: %s (HarrisZ+)" % (end - start))
        # show keypoints 
        to_save = iname + '_harrisz_plus_kornia.pdf'    
        print(f"Plot keypoint ellipses in Kornia and save the result in {to_save}")
        start = time.time()
        # show keypoints with Kornia
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)    
        visualize_LAF(K.image_to_tensor(img, False), lafs, 0)
        plt.axis('off')    
        # save the plot        
        plt.savefig(to_save, dpi=150, bbox_inches='tight')
        end = time.time()
        print("Elapsed time: %s (plot)" % (end - start))
        print("")
