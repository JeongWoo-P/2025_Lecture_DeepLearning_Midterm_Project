# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting


# [TODO: Optional] Rewrite this class if you want
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNetwork(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()

        # 초기 AlexNet 스타일 블록
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Residual Block 1: 192 → 384
        self.res_block1_conv1 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.res_block1_relu1 = nn.ReLU(inplace=True)
        self.res_block1_conv2 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.res_block1_relu2 = nn.ReLU(inplace=True)
        self.res_block1_proj = nn.Conv2d(192, 384, kernel_size=1)

        # Residual Block 2: 384 → 256
        self.res_block2_conv1 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.res_block2_relu1 = nn.ReLU(inplace=True)
        self.res_block2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res_block2_relu2 = nn.ReLU(inplace=True)
        self.res_block2_proj = nn.Conv2d(384, 256, kernel_size=1)

        # 마지막 풀링
        self.final_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # classifier (FC layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)

        # Residual Block 1
        identity1 = self.res_block1_proj(x)
        out = self.res_block1_relu1(self.res_block1_conv1(x))
        out = self.res_block1_conv2(out)
        x = self.res_block1_relu2(out + identity1)

        # Residual Block 2
        identity2 = self.res_block2_proj(x)
        out = self.res_block2_relu1(self.res_block2_conv1(x))
        out = self.res_block2_conv2(out)
        x = self.res_block2_relu2(out + identity2)

        x = self.final_pool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'MyNetwork',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork(num_classes=num_classes, dropout=0.5)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)

        self.log_dict({'loss/train': loss,
                        'accuracy/train': accuracy,
                        'f1/train': f1.mean()},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)
        self.log_dict({'loss/val': loss, 
                       'accuracy/val': accuracy,
                       'f1/val': f1.mean()},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
