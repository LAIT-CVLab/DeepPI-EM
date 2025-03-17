import numpy as np

import torch
import torchvision.transforms.functional as TF


class ToTensor(object):
    def __call__(self, data):
        label_, input_ = data['label'], data['input']
        
        label_ = TF.to_tensor(label_).type(torch.float32)
        input_ = TF.to_tensor(input_).type(torch.float32)
        
        data = {'label': label_, 'input': input_}
        
        return data


class RandomCrop(object):
    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size
    
    def __call__(self, data):
        label_, input_ = data['label'], data['input']
        
        h = input_.shape[1]
        w = input_.shape[2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        label_ = label_[:, top:top+new_h, left:left+new_w]
        input_ = input_[:, top:top+new_h, left:left+new_w]
        
        data = {'label': label_, 'input': input_}
        
        return data
    

class RandomFlip(object):
    def __call__(self, data):
        label_, input_ = data['label'], data['input']

        if np.random.rand() > 0.5:
            label_ = TF.hflip(label_)
            input_ = TF.hflip(input_)
            
        if np.random.rand() > 0.5:
            label_ = TF.vflip(label_)
            input_ = TF.vflip(input_)

        data = {'label': label_, 'input': input_}

        return data

    
class Resize(object):
    def __call__(self, data):
        label_, input_ = data['label'], data['input']

        _, label_h, label_w = label_.shape
        _, input_h, input_w = input_.shape
        
        max_size = 1280 
        
        def resize_tensor(tensor, h, w):
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                tensor = TF.resize(tensor, [new_h, new_w])
            return tensor

        label_ = resize_tensor(label_, label_h, label_w)
        input_ = resize_tensor(input_, input_h, input_w)
        
        data = {'label': label_, 'input': input_}
        
        return data
    

class Rotate(object):
    def __call__(self, data):
        label_, input_ = data['label'], data['input']
        
        angle = np.random.randint(-180, 180)
        while np.abs(angle) < 5:
            angle = np.random.randint(-90, 90)
            
        label_ = TF.rotate(label_, angle)
        input_ = TF.rotate(input_, angle)
      
        data = {'label': label_, 'input': input_}
        
        return data 