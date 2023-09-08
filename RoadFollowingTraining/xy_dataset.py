'''
////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2023 Mateusz Malinowski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////
'''

import glob
import os

import cv2
import numpy as np
import PIL.Image
import torch
import torch.utils.data
import torchvision.transforms


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, transform: torchvision.transforms.Compose = None, random_hflip: bool = False, greyscale: bool = False):
        super(XYDataset, self).__init__()
        self.directory = directory
        self.transform = transform
        self.random_hflip = random_hflip
        self.greyscale = greyscale
        self.refresh()
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_GRAYSCALE if self.greyscale else cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        
        # both steering and throttle are already from -1 to 1. 
        x = ann['x'] # -1 left, +1 right
        y = ann['y'] # -1 top, +1 bottom
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
            
        return image, torch.Tensor([x, y])
    
    def _parse(self, path: str):
        basename = os.path.basename(path)
        items = basename.split('_')
        x = items[0]
        y = items[1]
        return float(x), float(y)
        
    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'x': x,
                'y': y
            }]

