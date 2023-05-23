import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class MLP(pl.LightningModule):
    def __init__(self, input_size, lr=1e-3, binary=False):
        super().__init__()
        self.input_size = input_size
        self.lr = lr
        self.loss_function = F.binary_cross_entropy_with_logits if binary else F.mse_loss
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

    def compute_loss(self, batch):
        x = batch[0]
        y = batch[1].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = self.loss_function(x_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=8, factor=0.1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class ILModel(MLP):
    """
    Irreducible loss model for RHO-Loss
    """

    def compute_loss(self, batch):
        idx, x, y = batch
        loss = super().compute_loss((x, y))
        return loss


class RLossModel(MLP):
    """
    Reducible loss model for RHO-Loss
    """

    def __init__(self, input_size, selection_method=None, **kwargs):
        super().__init__(input_size, **kwargs)
        self.selection_method = selection_method

    def compute_loss(self, batch):
        idx, x, y = batch
        loss = super().compute_loss((x, y))
        return loss

    def training_step(self, batch, batch_idx):
        batch = self.selection_method(batch=batch, model=self, loss_function=self.loss_function)
        loss = super().training_step(batch, batch_idx)
        return loss
