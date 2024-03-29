# -*- coding:utf-8 -*-
import os
import argparse, os, sys, glob
from argparse import ArgumentParser
from omegaconf import OmegaConf
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append("/home/xiebaoye/latent-diffusion/cond_ldm/")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torchvision.utils import save_image
from pytorch_lightning.loggers import WandbLogger
from taming.modules.vqvae.quantize import VectorQuantizer as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from utils.util import instantiate_from_config
from base.init_experiment import initExperiment_v2
import numpy as np

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='vqgan_uncond_gloss_total')
    parser.add_argument('--result_root', type=str, default=r'/xxx/xxx/vqgan_train/logs')
    parser.add_argument('--data_config',
                        default=r'/xxx/xxx/datasets.yaml')
    parser.add_argument("--command", default="fit")
    parser.add_argument('--devices', default=[3])
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--limit_train_batches", type=int, default=100)
    parser.add_argument("--base_learning_rate", type=float, default=2.5e-5) # 4.5e-6
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--scale_lr', type=bool, default=True)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def main(opts, data_cfg):

    datamodule = instantiate_from_config(data_cfg)
    model = VQModel(opts)
    if opts.command == "fit":
        #wandb_logger = WandbLogger(project="vqgan_rec", name=opts.exp_name, save_dir=opts.default_root_dir)
        wandb_logger = None
        ckpt_callback = ModelCheckpoint(save_last=True, filename="model-{epoch}")
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=5)],
                                                logger=wandb_logger)
        # first step train (only l1loss)
        #trainer.fit(model=model, datamodule=datamodule)


        # second step train (edit disc_start = 1 )
        ckpt = '/xxx/xxx/vqgan——pretrain/last.ckpt'
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

class VQModel(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        ddconfig = {'double_z': False, 'z_channels': 4, 'resolution': 256, 'in_channels': 1, 'out_ch': 1, 'ch': 128,
                    'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        # 400 200 100 50 25
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        lossconfig = dict(disc_conditional=False, disc_in_channels=1, disc_start=1, disc_weight=0.6,
                          codebook_weight=1.0)
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)

        self.embed_dim = ddconfig["z_channels"]
        n_embed = 1024 # 16384 # 4 * 64 * 64 # latent 4 * 110 * 64
        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        self.lr_g_factor = 1.0

        self.save_hyperparameters()


    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch):
        x = batch['image']
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        if batch_idx == 0:
            self.batch_sample = batch

        x = self.get_input(batch)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def on_train_epoch_end(self):
        img_save_dir = os.path.join(self.opts.default_root_dir, 'test_results', str(self.current_epoch))
        os.makedirs(img_save_dir, exist_ok=True)
        with torch.no_grad():
            x = self.get_input(self.batch_sample)
            xrec, qloss, ind = self(x, return_pred_indices=True)
            x = x.cpu()
            xrec = xrec.cpu()
            for i in range(x.shape[0]):
                save_image([x[i] * 0.5 + 0.5, xrec[i] * 0.5 + 0.5],
                           os.path.join(img_save_dir, str(i) + '.png'))

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def configure_optimizers(self):
        lr_d = self.opts.learning_rate
        lr_g = self.lr_g_factor * self.opts.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    data_cfg = OmegaConf.load(opts.data_config)
    initExperiment_v2(opts, data_cfg)
    main(opts, data_cfg)

