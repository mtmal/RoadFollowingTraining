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

import argparse
import pathlib

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch_tensorrt
import torchvision
import torchvision.transforms as transforms
import tqdm

from xy_dataset import XYDataset


class Trainer:
    def __init__(self, directory: str, batch_size: int = 32, greyscale: bool = True) -> None:
        trans = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.greyscale = greyscale
        if greyscale:
            trans.transforms.append(transforms.Normalize([0.445], [0.269], inplace=True))
        else:
            trans.transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # initialise resnet 18 with one channel input data and two outputs (steering and throttle)
        self.model = torchvision.models.resnet18(pretrained=True)
        if greyscale:
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = torch.nn.Linear(512, 2)

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # prepare datasets
        dataset = XYDataset(directory, trans, random_hflip=True, greyscale=greyscale)
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
        self.eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

    def train(self, epochs: int = 5):
        self.model = self._train_eval(self.model.train(), self.train_loader, self.optimizer, epochs)

    def evaluate(self):
        self._train_eval(self.model.eval(), self.eval_loader)

    def save_model(self, path: str = "your_model.pth"):
        torch.save(self.model.state_dict(), path)

    def _train_eval(self, model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer = None, epochs: int = 1):
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
                sum_loss += float(loss)

            # Update the progress bar with the progress and current loss
            progress_bar.update()
            progress_bar.set_postfix(loss=sum_loss / i)

        torch.cuda.empty_cache()
        return model

    def optimise_model(self, path: str = "your_model.ts'"):
        if self.greyscale:
            inputs = [torch_tensorrt.Input(
                min_shape=[2, 1, 224, 224],
                opt_shape=[2, 1, 224, 224],
                max_shape=[4, 1, 224, 224],
                dtype=torch.half)]
        else:
            inputs = [torch_tensorrt.Input(
                min_shape=[1, 3, 224, 224],
                opt_shape=[1, 3, 224, 224],
                max_shape=[2, 3, 224, 224],
                dtype=torch.half)]

        trt_ts_module = torch_tensorrt.compile(self.model.cuda().eval().half(), inputs=inputs, enabled_precisions={torch.half})
        torch.jit.save(trt_ts_module, path)
        return trt_ts_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains resnet18 on images for steering and throttle.')
    parser.add_argument('--folder', type=pathlib.Path, default="/home/jetson/data/stereo/1694165232.458000/combined",
                        help='Path to folder with jpg images.')
    parser.add_argument('--epochs', type=int, default=5, help="The number of epochs to train.")
    parser.add_argument('--batch', type=int, default=32, help="The size of batch.")
    parser.add_argument('--greyscale', type=bool, default=True, help="Flag to indicate if input images will be greyscale or not.")
    parser.add_argument('--output_pt', type=pathlib.Path, default='your_model.pth', help='The name of Pytorch model to save.')
    parser.add_argument('--output_trt', type=pathlib.Path, default='your_model.ts', help='The name of optimised torch_tensorrt model to save.')
    args = parser.parse_args()
    
    print("Initialiasing class")
    trainer = Trainer(args.folder, args.batch, args.greyscale)
    print("Starting training")
    trainer.train(args.epochs)
    print("Running evaluation")
    trainer.evaluate()
    print("Saving unoptimised model")
    trainer.save_model(args.output_pt)
    print("Optimising model")
    trainer.optimise_model(args.output_trt)
