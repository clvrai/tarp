import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import kornia
from tarp.components.transform_layer import ColorJitterLayer


def random_crop(imgs, out=64):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3
    
    imgs = imgs.view([b,frames,3,h,w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114 
    
    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(device) # broadcast tiling
    return imgs

def random_grayscale(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or cuda
        returns torch.tensor
    """
    device = images.device
    in_type = images.type()
    images = images * 255.
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.
    return out

# random cutout
# TODO: should mask this 

def random_cutout(imgs, min_cut=4,max_cut=12):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """

    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts

def random_cutout_color(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """
    
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        
        # add random box
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
            rand_box[i].reshape(-1,1,1),                                                
            (1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])
        
        cutouts[i] = cut_img
    return cutouts

# random flip

def random_flip(images,p=.2):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or gpu, 
        p: prob of applying aug,
        returns torch.tensor
    """
    # images: [B, C, H, W]
    device = images.device
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    flipped_images = images.flip([3])
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] #// 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]
    
    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out

# random rotation

def random_rotation(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: str, cpu or gpu, 
        p: float, prob of applying aug,
        returns torch.tensor
    """
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    rot90_images = images.rot90(1,[2,3])
    rot180_images = images.rot90(2,[2,3])
    rot270_images = images.rot90(3,[2,3])    
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)
    
    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i,m in enumerate(masks):
        m[torch.where(mask==i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
        m = m[:,:,None,None]
        masks[i] = m
    
    
    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([bs, -1, h, w])
    return out


# random color

    

def random_convolution(imgs):
    '''
    random covolution in "network randomization"
    
    (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
    '''
    _device = imgs.device
    
    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)
    
    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
    
    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_imgs = imgs[trans_index*batch_size:(trans_index+1)*batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return total_out


def random_color_jitter(imgs):
    """
        inputs np array outputs tensor
    """
    b,c,h,w = imgs.shape
    imgs = imgs.view(-1,3,h,w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, 
                                                contrast=0.4,
                                                saturation=0.4, 
                                                hue=0.5, 
                                                p=1.0, 
                                                batch_size=128))

    imgs = transform_module(imgs).view(b,c,h,w)
    return imgs

class RandomShift(nn.Module):
    def __init__(self, resolution=64, random_shift_pad=4):
        super().__init__()
        self.trans = nn.Sequential(
                    nn.ReplicationPad2d(random_shift_pad),
                    kornia.augmentation.RandomCrop((resolution, resolution))
                )
    def forward(self, imgs):
        return self.trans(imgs)

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


def no_aug(x):
    return x

def numpify(func):
    """Wrapper so that the augmentation function always works on a numpy
    array, but if the input `imgs` is a torch tensor, a torch tensor will be
    returned. Assumes first input and first output of the function is the
    images array/tensor, and only operates on that."""
    def numpified_aug(imgs, *args, **kwargs):
        _numpify = isinstance(imgs, torch.Tensor)
        if _numpify:
            imgs = imgs.cpu().numpy()
        ret = func(imgs, *args, **kwargs)
        if _numpify:
            if isinstance(ret, tuple):
                # Assume first is the augmented images.
                ret = (torch.from_numpy(ret[0]), *ret[1:])
            else:
                ret = torch.from_numpy(ret)
        return ret
    return numpified_aug


##############################################################################
# SHIFTS
##############################################################################


@numpify
def subpixel_shift(imgs, max_shift=1.):
    """
    Pad input images by 1 using "edge" mode, and then do a nearest-neighbor
    averaging scheme, centered at a random location for each image, up to
    max_shift away from the origin in each x and y.  Each output pixel will
    be a linear interpolation of the surrounding 2x2 input pixels.
    """
    if imgs.dtype == np.uint8:
        raise NotImplementedError
    assert max_shift <= 1.
    b, c, h, w = imgs.shape
    padded = np.pad(
        imgs,
        pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="edge",
    )
    xx = np.array([[-1., 0., 1.]])  # [1,3]

    rand_x = max_shift * (2 * np.random.rand(b, 1) - 1)  # [B,1]
    rand_y = max_shift * (2 * np.random.rand(b, 1) - 1)  # [B,1]

    wx = np.maximum(0., 1 - np.abs(xx - rand_x))  # [B,3]
    wy = np.maximum(0., 1 - np.abs(xx - rand_y))  # [B,3]
    weight = wx.reshape(b, 1, 3) * wy.reshape(b, 3, 1)  # [B,1,3]x[B,3,1]->[B,3,3]

    shifted = np.zeros_like(imgs)
    for dy in [0, 1, 2]:
        for dx in [0, 1, 2]:
            shifted += (weight[:, dy, dx].reshape(-1, 1, 1, 1) *
                padded[:, :, dy:h + dy, dx:w + dx])

    return shifted

@numpify
def random_shift(imgs, pad=1, prob=1.):
    t = b = c = 1
    shape_len = len(imgs.shape)
    if shape_len == 2:  # Could also make all this logic into a wrapper.
        h, w = imgs.shape
    elif shape_len == 3:
        c, h, w = imgs.shape
    elif shape_len == 4:
        b, c, h, w = imgs.shape
    elif shape_len == 5:
        t, b, c, h, w = imgs.shape  # Apply same crop to all T
        imgs = imgs.transpose(1, 0, 2, 3, 4)
        _c = c
        c = t * c
        # imgs = imgs.reshape(b, t * c, h, w)
    imgs = imgs.reshape(b, c, h, w)

    crop_h = h
    crop_w = w

    padded = np.pad(
        imgs,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="edge",
    )
    b, c, h, w = padded.shape

    h_max = h - crop_h + 1
    w_max = w - crop_w + 1
    h1s = np.random.randint(0, h_max, b)
    w1s = np.random.randint(0, w_max, b)
    if prob < 1.:
        which_no_crop = np.random.rand(b) > prob
        h1s[which_no_crop] = pad
        w1s[which_no_crop] = pad

    shifted = np.zeros_like(imgs)
    for i, (pad_img, h1, w1) in enumerate(zip(padded, h1s, w1s)):
        shifted[i] = pad_img[:, h1:h1 + crop_h, w1:w1 + crop_w]

    if shape_len == 2:
        shifted = shifted.reshape(crop_h, crop_w)
    elif shape_len == 3:
        shifted = shifted.reshape(c, crop_h, crop_w)
    elif shape_len == 5:
        shifted = shifted.reshape(b, t, _c, crop_h, crop_w)
        shifted = shifted.transpose(1, 0, 2, 3, 4)

    return shifted

if __name__ == '__main__':
    import time 
    from tabulate import tabulate
    def now():
        return time.time()
    def secs(t):
        s = now() - t
        tot = round((1e5 * s)/60,1)
        return round(s,3),tot

    x = np.load('data_sample.npy',allow_pickle=True)
    x = np.concatenate([x,x,x],1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.from_numpy(x).to(device)
    x = x.float() / 255.

    # crop
    t = now()
    random_crop(x.cpu().numpy(),64)
    s1,tot1 = secs(t)
    # grayscale 
    t = now()
    random_grayscale(x,p=.5)
    s2,tot2 = secs(t)
    # normal cutout 
    t = now()
    random_cutout(x.cpu().numpy(),10,30)
    s3,tot3 = secs(t)
    # color cutout 
    t = now()
    random_cutout_color(x.cpu().numpy(),10,30)
    s4,tot4 = secs(t)
    # flip 
    t = now()
    random_flip(x,p=.5)
    s5,tot5 = secs(t)
    # rotate 
    t = now()
    random_rotation(x,p=.5)
    s6,tot6 = secs(t)
    # rand conv 
    t = now()
    random_convolution(x)
    s7,tot7 = secs(t)
    # rand color jitter 
    t = now()
    random_color_jitter(x)
    s8,tot8 = secs(t)
    
    print(tabulate([['Crop', s1,tot1], 
                    ['Grayscale', s2,tot2], 
                    ['Normal Cutout', s3,tot3], 
                    ['Color Cutout', s4,tot4], 
                    ['Flip', s5,tot5], 
                    ['Rotate', s6,tot6], 
                    ['Rand Conv', s7,tot7], 
                    ['Color Jitter', s8,tot8]], 
                    headers=['Data Aug', 'Time / batch (secs)', 'Time / 100k steps (mins)']))
