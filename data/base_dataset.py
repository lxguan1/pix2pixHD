import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from scipy import interpolate

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, norm_val = 0.5):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Lambda(lambda img: normalize_img(img, norm_val))] #[transforms.Normalize((0.5), #
            #                                    (0.5))]
    return transforms.Compose(transform_list)

def normalize(): #return a new transform that normalizes the img   
    return transforms.Normalize((0.5), (0.5))

def normalize_img(img, norm_val): #Need to determine which normalization factor
    return img / norm_val

def __make_power_2(img, base, method=Image.BICUBIC):
    #print('one', flush = True)
    ow = img.shape[1] 
    oh = img.shape[0] 
    range_x = np.linspace(0, ow, ow)
    range_y = np.linspace(0, oh, oh)
    range_x_mesh, range_y_mesh = np.meshgrid(range_x, range_y)
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    f = interpolate.interp2d(range_x_mesh, range_y_mesh, img[:,:,0], kind='cubic')
    
    return f(w, h)[:,:,np.newaxis] 

def __scale_width(img, target_width, method=Image.BICUBIC): #NOT USED
    ow = img.shape[1]
    oh = img.shape[0]
    #print('one', flush = True)
    range_x = np.linspace(0, ow, ow)
    range_y = np.linspace(0, oh, oh)
    range_x_mesh, range_y_mesh = np.meshgrid(range_x, range_y)
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow) 
    f = interpolate.interp2d(range_x_mesh, range_y_mesh, img[:,:,0], kind='cubic')
    print('one scale_width', flush=True)
    return f(np.arange(w), np.arange(h))[:,:,np.newaxis]

def __crop(img, pos, size): #Used
    ow= img.shape[1] 
    oh = img.shape[0]
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img[y1: y1 + th, x1:x1 + tw]
    return img

def __flip(img, flip):
    if flip:
        return np.fliplr(img).copy() 
    return img
