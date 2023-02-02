"""
Training scripts for uni/multimodal transformers with TPR head attached.
Configured to train on ImageNet1/21k
"""


import imgaug
import numpy as np
import pytorch_lightning as pl
import torchvision

from imgaug import augmenters as iaa
from torch.utils.data import DataLoader

from datasets import ImageNet
from models.tpr_block_vit_lightning import PLModel


def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)


def main():
    # train_ds_path = r'B:\Datasets\imagenet21k_resized\imagenet21k_train'
    # val_ds_path = r'B:\Datasets\imagenet21k_resized\imagenet21k_val'
    # train_ds_path = r'/media/weiyuen/SSD/Datasets/imagenet21k_resized/imagenet21k_train'
    # val_ds_path = r'/media/weiyuen/SSD/Datasets/imagenet21k_resized/imagenet21k_val'
    train_ds_path = r'/media/weiyuen/SSD/Datasets/ImageNet2/train'
    val_ds_path = r'/media/weiyuen/SSD/Datasets/ImageNet2/validation'

    # hardware parameters
    n_workers = 12
    backend = 'nccl'  # gloo for Win, nccl for Linux
    
    # model parameters
    num_classes = 1000
    dim = 768
    heads = 12
    mlp_dim = 3072
    n_roles = 12
    dim_head = 64
    tpr_dim_head = 64
    freeze_encoder = False

    # training parameters
    # pretrained should be False or path to ckpt file
    pretrained = False
    batch_size = 128
    epochs = 64
    lr = 1e-4
    label_smoothing = 0.1

    wandb_logger = pl.loggers.WandbLogger(
        project="tpr-block-vit", log_model=True
    )

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
        train_ds_path,
        transform=transform
    )
    valid_ds = ImageNet(
        val_ds_path,
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
        pretrained=pretrained,
        lr=lr,
        epochs=epochs,
        label_smoothing=label_smoothing,
        num_classes=num_classes,
        dim=dim,
        heads=heads,
        mlp_dim=mlp_dim,
        n_roles=n_roles,
        dim_head=dim_head,
        tpr_dim_head=tpr_dim_head,
        freeze_encoder=freeze_encoder
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
        callbacks=[ckpt_callback, lr_monitor],  # add swa here if necessary
        precision=16,
        track_grad_norm=2,
        strategy=pl.strategies.DDPStrategy(
            find_unused_parameters=True,
            process_group_backend=backend
        )
    )
    ckpt_path = r'tpr-block-vit/2yih576l/checkpoints/epoch=27-step=35056.ckpt'
    # when resuming from checkpoint
    trainer.fit(model, train_datagen, valid_datagen, ckpt_path=ckpt_path)

    # when training from scratch
    # trainer.fit(model, train_datagen, valid_datagen)


if __name__ == '__main__':
    main()
