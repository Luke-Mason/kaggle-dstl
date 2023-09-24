from pathlib import Path

import torch
from torch import nn, optim
from torch.autograd import Variable
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
        # where X is L1Loss, MSELoss, CrossEntropyLoss
        # CTCLoss, NLLLoss, PoissonNLLLoss,
        # KLDivLoss, BCELoss, BCEWithLogitsLoss,
        # MarginRankingLoss, HingeEmbeddingLoss,
        # MultiLabelMarginLoss, SmoothL1Loss,
        # SoftMarginLoss, MultiLabelSoftMarginLoss,
        # CosineEmbeddingLoss, MultiMarginLoss,
        # or TripletMarginLoss
        self.mse_loss = nn.MSELoss()
        self.optimizer: optim.Optimizer or None = None
        self.tb_logger: SummaryWriter or None = None
        self.logdir: Path or None = None
        self.on_gpu = torch.cuda.is_available()
        if self.on_gpu:
            self.nnetwork.cuda()

    # TODO Make each batch a set of random separate images
    # TODO Make each image go through network and compare the salient pixels
    #  against the ground truth mask.
    # TODO compare each bounding box with IoU, dice loss, jaccard. Add it all
    #  to a total loss.

    def loss(self, outputs, targets):
        return self.bce_loss(outputs, targets)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader):
        self.optimizer = optim.Adam(self.nnetwork.parameters(), lr=self.hps.lr,
                                    weight_decay=self.hps.weight_decay)
        comment = f' batch_size = {self.hps.batch_size} lr = {self.hps.lr} weight_decay = {self.hps.weight_decay}'
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

            # Loss
            # loss = self.loss(outputs, targets)
            # loss.backward()

            batch_size = x.size()[0]
            losses = self.class_losses(y, dist_y, y_pred)
            total_loss = losses[0]
            for l in losses[1:]:
                total_loss += l
            (total_loss * batch_size).backward()

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
                loss += self.loss(outputs, targets).item()
        loss /= len(valid_loader)
        self.tb_logger.add_scalar('valid/loss', loss, epoch)
        print('\nValidation set: Average loss: {:.4f}\n'.format(loss))

    def save_model(self, epoch):
        torch.save(self.nnetwork.state_dict(), self.logdir / f'epoch_{epoch}.pth')

    def load_model(self, path):
        self.nnetwork.load_state_dict(torch.load(path))

    def predict(self, inputs):
        self.nnetwork.eval()
        with torch.no_grad():
            if self.on_gpu:
                inputs = inputs.cuda()
            outputs = self.nnetwork(inputs)
        return outputs

    def class_losses(self,
                     ys: torch.FloatTensor,
                     ys_dist: torch.FloatTensor,
                     y_preds: Variable):

        if self.on_gpu:
            ys = ys.cuda()


        class_losses = []
        ys = self._var(ys)
        if self.hps.needs_dist:
            ys_dist = self._var(ys_dist)
        for cls_idx, _ in enumerate(self.hps.classes):
            y, y_pred = ys[:, cls_idx], y_preds[:, cls_idx]
            y_dist = ys_dist[:, cls_idx] if self.hps.needs_dist else None
            loss = self.calc_loss(y, y_dist, y_pred)
            class_losses.append(loss)
        return class_losses

    def calc_loss(self, y, y_dist, y_pred):
        hps = self.hps
        loss = 0.
        if hps.log_loss:
            loss += self.bce_loss(y_pred, y) * hps.log_loss
        if hps.dice_loss:
            intersection = (y_pred * y).sum()
            union_without_intersection = y_pred.sum() + y.sum()  # without intersection union

            if union_without_intersection[0] != 0:
                loss += (
                                    1 - intersection / union_without_intersection) * hps.dice_loss

        if hps.jaccard_loss:
            intersection = (y_pred * y).sum()
            union = y_pred.sum() + y.sum() - intersection

            if union[0] != 0:
                loss += (1 - intersection / union) * hps.jaccard_loss

        if hps.dist_loss:
            loss += self.mse_loss(y_pred, y_dist) * hps.dist_loss

        if hps.dist_dice_loss:
            intersection = (y_pred * y_dist).sum()
            union_without_intersection = y_pred.sum() + y_dist.sum()  # without intersection union
            if union_without_intersection[0] != 0:
                loss += (
                                    1 - intersection / union_without_intersection) * hps.dist_dice_loss

        if hps.dist_jaccard_loss:
            intersection = (y_pred * y_dist).sum()
            union = y_pred.sum() + y_dist.sum() - intersection
            if union[0] != 0:
                loss += (1 - intersection / union) * hps.dist_jaccard_loss

        loss /= (hps.log_loss + hps.dist_loss + hps.dist_jaccard_loss +
                 hps.dist_dice_loss + hps.dice_loss + hps.jaccard_loss)
        return loss

class SatelliteImageDataset(Dataset):
    def __init__(self, data_paths, transform):
        self.data_paths = data_paths
        self.transform = transform

    # Length of the dataset
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image = Image.open(self.data_paths[idx])
        image = self.transform(image)
        return image
