"""
TPR Block ViT with HF base wrapped in LightningModule.
Allows for transfer learning from ImageNet21k -> 1k
"""


import pytorch_lightning as pl
import torch
import torchmetrics

from torch import nn

from .tpr_block_vit_hf import TPViT


class PLModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        pretrained=False,
        lr=0.1,
        epochs=100,
        label_smoothing=0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs
        self.model = TPViT(**kwargs)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.softmax = torch.nn.functional.softmax
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes)
        self.top_k_acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes, top_k=5)

        if pretrained is not False:
            # Filter out MLP layer weights
            state_dict = torch.load(pretrained)['state_dict']
            filtered_state_dict = {k: v for (k, v) in state_dict.items() if 'mlp_head' not in k}
            filtered_state_dict = {k[6:]: v for (k, v) in filtered_state_dict.items()}  # remove model. in name
            self.model.load_state_dict(filtered_state_dict)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(kwargs['dim']),
            nn.Linear(kwargs['dim'], num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return self.mlp_head(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        acc = self.acc(self.softmax(outputs, dim=1), y)
        top_k_acc = self.top_k_acc(self.softmax(outputs, dim=1), y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_acc_top_k', top_k_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        acc = self.acc(self.softmax(outputs, dim=1), y)
        top_k_acc = self.top_k_acc(self.softmax(outputs, dim=1), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc_top_k', top_k_acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        acc = self.acc(self.softmax(outputs, dim=1), y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            # momentum=0.9,
            weight_decay=0.3
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=8,
                factor=0.1,
                min_lr=1e-7
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "name": "lr"
        }
        return [optimizer], [scheduler]
