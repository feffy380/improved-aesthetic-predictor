import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class MLP(pl.LightningModule):
    def __init__(self, input_size, lr=1e-3):
        super().__init__()
        self.input_size = input_size
        self.lr = lr
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=10, factor=0.1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class ILModel(MLP):
    def training_step(self, batch, batch_idx):
        idx, x, y = batch
        y = y.reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        idx, x, y = batch
        y = y.reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss


class RLossModel(MLP):
    def __init__(self, input_size, lr=1e-3, selection_method=None):
        super().__init__(input_size, lr)
        self.selection_method = selection_method

    def training_step(self, batch, batch_idx):
        batch = self.selection_method(batch=batch, model=self, loss_function=F.mse_loss)
        idx, x, y = batch
        y = y.reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        idx, x, y = batch
        y = y.reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
