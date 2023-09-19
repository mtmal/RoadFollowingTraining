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

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm

from xy_dataset import XYDataset


class Trainer:
    def __init__(self, directory: str, batch_size: int = 16, greyscale: bool = True) -> None:
#        self.greyscale = greyscale
        if greyscale:
            trans = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.445], [0.269], inplace=True)
            ])
        else:
            trans = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
            ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # initialise resnet 18 with one channel input data and two outputs (steering and throttle)
        self.model = torchvision.models.resnet18(pretrained=True)
#        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        if greyscale:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = torch.nn.Linear(512, 2)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        # prepare datasets
        dataset = XYDataset(directory, trans, random_hflip=True, greyscale=greyscale)
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
        self.eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
        
    def train(self, epochs: int = 5):
        print("Training")
        self.model = self._train_eval(self.model.train(), self.train_loader, self.optimizer, epochs)
        
    def evaluate(self):
        print("Evaluation")
        self._train_eval(self.model.eval(), self.eval_loader)
        
    def save_model(self, path: str = 'your_model.pth'):
        torch.save(self.model.state_dict(), path)
        
    def _train_eval(self, model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer = None, epochs: int = 1):
        progress_bar = tqdm.tqdm(total=epochs, desc="Training Progress in epochs" if model.training else "Validation Progress in epochs")

        for _ in range(0, epochs):
            i = 0
            sum_loss = 0.0
            
            for images, xy in iter(dataloader):
                if model.training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images.to(self.device))
                # compute MSE loss over x, y coordinates
                loss = torch.mean(torch.pow(outputs - xy.to(self.device), 2))

                if model.training:
                    # run backpropogation to accumulate gradients
                    loss.backward()
                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                i += len(xy)
                sum_loss += float(loss.to(torch.device('cpu')))
                
            # Update the progress bar with the progress and current loss
            progress_bar.update()
            progress_bar.set_postfix(loss=sum_loss / i)
                
        model = model.eval()
        torch.cuda.empty_cache()
        return model

if __name__ == "__main__":
    trainer = Trainer(directory="/home/mati/data/stereo/1694165232.458000/combined")
    trainer.train()
    trainer.save_model()
    trainer.evaluate()

