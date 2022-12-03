"""
Training scripts for uni/multimodal transformers with TPR head attached.
Configured to train on ImageNet1/21k
"""


import imgaug
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision

from imgaug import augmenters as iaa
from torch.utils.data import DataLoader

from datasets import ImageNet
# from models.tpr_block_vit_lucidrains import TPViT
from models.tpr_block_vit_hf import TPViT


class PLModel(pl.LightningModule):
    def __init__(self, lr=0.1, epochs=100, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs
        self.model = TPViT(
            # image_size=224,
            # patch_size=16,
            num_classes=1000,
            dim=768,
            # depth=10,
            heads=8,
            mlp_dim=3072,
            n_roles=8,
            dim_head=96,
            freeze_encoder=False
        )
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=kwargs['label_smoothing'])
        self.softmax = torch.nn.functional.softmax
        self.acc = torchmetrics.Accuracy()
        self.top_k_acc = torchmetrics.Accuracy(top_k=5)
        
    def forward(self, x):
        return self.model(x)
        
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
            weight_decay=0.01
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=5,
                factor=0.1,
                min_lr=5e-7
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "name": "lr"
        }
        return [optimizer], [scheduler]


def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)


def main():
    # hardware parameters
    n_workers = 8
    
    # model parameters
    image_size = 224
    patch_size = 16
    num_classes = 1000
    dim = 768
    depth = 12
    heads = 12
    mlp_dim = 3072
    dropout = 0.1

    # training parameters
    batch_size = 128
    epochs = 128
    lr = 5e-4
    label_smoothing = 0.1

    wandb_logger = pl.loggers.WandbLogger(
        project="tpr-block-vit", log_model=True
    )

    '''transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.1),
        torchvision.transforms.RandomAffine(degrees=(-15, 15), translate=(0.15, 0.15), shear=5),
        torchvision.transforms.ToTensor()
    ])
    
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])'''
    transform = torchvision.transforms.Compose([
        iaa.Sequential([
            iaa.Resize({"height": 224, "width": 224}),
            # iaa.CropToFixedSize(width=224, height=224),
            iaa.flip.Fliplr(p=0.5),
            iaa.Affine(
                scale=(0.8, 1.2),
                translate_percent=0.2,
                rotate=(-20, 20),
                shear=(-5, 5)
            ),
            iaa.AdditiveGaussianNoise((0, 15), per_channel=True),
            iaa.GaussianBlur(sigma=(0.0, 0.15)),
            iaa.MultiplyBrightness(mul=(0.6, 1.4)),
            iaa.MultiplyHueAndSaturation((0.6, 1.4), per_channel=True),
            iaa.GammaContrast(),
            # iaa.ChangeColorTemperature((2000, 10000)),
            iaa.Cutout(nb_iterations=(0,2), size=0.2),
            iaa.Dropout((0, 0.1))
        ]).augment_image,
        torchvision.transforms.ToTensor()
    ])
    val_transform = torchvision.transforms.Compose([
        iaa.Sequential([
            iaa.Resize({"height": 224, "width": 224}),
        ]).augment_image,
        torchvision.transforms.ToTensor()
    ])

    '''train_ds = torchvision.datasets.ImageFolder(
        r'B:\Datasets\ImageNet2\train',
        transform=transform
    )
    valid_ds = torchvision.datasets.ImageFolder(
        r'B:\Datasets\ImageNet2\validation',
        transform=val_transform
    )'''

    train_ds = ImageNet(
        r'B:\Datasets\ImageNet2\train',
        transform=transform
    )
    valid_ds = ImageNet(
        r'B:\Datasets\ImageNet2\validation',
        transform=val_transform
    )

    train_datagen = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    valid_datagen = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    model = PLModel(
        lr=lr,
        epochs=epochs,
        label_smoothing=label_smoothing
    )
    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    swa = pl.callbacks.StochasticWeightAveraging(swa_lrs=5e-2)

    trainer = pl.Trainer(
        gradient_clip_val=1,
        # accumulate_grad_batches=16,
        logger=wandb_logger,
        log_every_n_steps=25,
        accelerator='gpu',
        devices=2,
        max_epochs=epochs,
        callbacks=[ckpt_callback, lr_monitor, swa],
        precision=16,
        track_grad_norm=2,
        strategy=pl.strategies.DDPStrategy(
            find_unused_parameters=True,
            process_group_backend='gloo'
        )
    )
    ckpt_path = r'tpr-block-vit\pixybg2w\checkpoints\epoch=23-step=120120.ckpt'
    # when resuming from checkpoint
    trainer.fit(model, train_datagen, valid_datagen, ckpt_path=ckpt_path)

    # when training from scratch
    # trainer.fit(model, train_datagen, valid_datagen)


if __name__ == '__main__':
    main()
