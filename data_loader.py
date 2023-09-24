from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import networks
from HyperParams import HyperParams
class Model:
    def __init__(self, hps: HyperParams):
        self.hps = hps
        self.nnetwork = getattr(networks, hps.net)(hps)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.optimizer = None  # type: optim.Optimizer
        self.tb_logger = None  # type: SummaryWriter
        self.logdir = None  # type: Path
        self.on_gpu = torch.cuda.is_available()
        if self.on_gpu:
            self.nnetwork.cuda()

    def loss(self, outputs, targets):
        return self.bce_loss(outputs, targets)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader):
        self.optimizer = optim.Adam(self.nnetwork.parameters(), lr=self.hps.lr)
        comment = f' batch_size = {self.hps.batch_size} lr = {self.hps.lr}'
        self.tb_logger = SummaryWriter(comment=comment)
        for epoch in range(self.hps.epochs):
            self.train_epoch(epoch, train_loader)
            if valid_loader:
                self.validate(epoch, valid_loader)
            self.save_model(epoch)
        self.tb_logger.close()

    def train_epoch(self, epoch, train_loader):
        self.nnetwork.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if self.on_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.nnetwork(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.tb_logger.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def validate(self, epoch, valid_loader):
        self.nnetwork.eval()
        loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                if self.on_gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                outputs = self.nnetwork(inputs)
                loss += self.bce_loss(outputs, targets).item()
        loss /= len(valid_loader)
        self.tb_logger.add_scalar('valid/loss', loss, epoch)
        print('\nValidation set: Average loss: {:.4f}\n'.format(loss))

    def save_model(self, epoch):
        torch.save(self.nnetwork.state_dict(), self.logdir / f'epoch_{epoch}.pth')

class SatelliteImageDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image = Image.open(self.data_paths[idx])
        image = self.transform(image)
        return image




# Create the dataset and dataloader
# Sample list of data paths (replace with actual file paths)
data_paths = ['path_to_image1.jpg', 'path_to_image2.jpg', 'path_to_image3.jpg']
batch_size = 32
num_epochs = 10

