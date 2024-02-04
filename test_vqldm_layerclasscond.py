# -*- coding:utf-8 -*-
import os
import argparse, os, sys, glob
from argparse import ArgumentParser
from omegaconf import OmegaConf
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append("/home/xiebaoye/latent-diffusion/cond_ldm/")
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_lightning.loggers import WandbLogger
from ldm.models.diffusion.ddim import DDIMSampler
from base.init_experiment import initExperiment_v2
from utils.util_for_opencv_diffusion import DDPM_base, disabled_train
from ldm.lr_scheduler import LambdaLinearScheduler
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.util import default
from imggen.train_vqgan_uncond import VQModel
from networks.openaimodel_maskclasscond_ldm import UNetModel
from train_vqldm_layerclasscond import LDM


def get_parser():
    parser = ArgumentParser()

    # seven
    parser.add_argument("--result_save_dir", type=str,
                        default='/home/Data/xiebaoye/train_logs/diffusion_results/cldm_res/vqldm_2d_layerClassCond_seven_balanced_2023-09-20T21-45-28/test_res/')
    # seven
    parser.add_argument('--ckpt_path', type=str,
                        default='/home/Data/xiebaoye/cldm_train_logs/ldm_train/logs/vqldm_2d_layerClassCond_seven_balanced_2023-09-20T21-45-28/layer2img/xcpmd624/checkpoints/model-epoch=29.ckpt')

    # seven
    parser.add_argument('--data_config',
                        default=r'/home/xiebaoye/latent-diffusion/cond_ldm/configs/layer-latent-diffusion/100edema_srf_seven.yaml')

    parser.add_argument('--first_stage_ckpt', type=str,
                        default='/home/Data/xiebaoye/train_logs/cldm_train_logs/vqgan_train/logs/vqgan_uncond_gloss_total_2023-08-14T10-30-12/checkpoints/last.ckpt')



    parser.add_argument("--image_size", default=(480, 256))
    parser.add_argument("--latent_size", default=(120, 64))
    parser.add_argument("--latent_channel", default=4)

    parser.add_argument('--use_ddim', type=bool, default=True)
    parser.add_argument('--ddim_steps', type=int, default=200)

    parser.add_argument("--command", default="test")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=[2])
    parser.add_argument('--reproduce', type=int, default=False)

    return parser


def main(opts):
    for k, v in opts.__dict__.items():print(f"{k}: {v}")

    data_cfg = OmegaConf.load(opts.data_config)
    data = instantiate_from_config(data_cfg)
    model = LDM_test(opts)
    trainer = pl.Trainer.from_argparse_args(opts)

    trainer.test(model=model, datamodule=data)

    # data.setup()
    # dataloader = data.train_dataloader()
    #
    # from tqdm import tqdm
    # from matplotlib import pylab as plt
    # import numpy as np
    #
    # for i, batch in enumerate(tqdm(dataloader)):
    #     xrec = model.test_step(batch, 0)
    #
    #     def imshow(img):
    #         # img = img / 2 + 0.5  # unnormalize
    #         npimg = img.detach().numpy()
    #         npimg = (npimg + 1.0) / 2.0
    #         npimg = (npimg * 255).astype(np.uint8)
    #         plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')  # 将【3，32，128】-->【32,128,3】
    #         plt.axis('off')
    #         plt.show()
    #
    #     images = torch.cat([batch['image'], xrec], 0)
    #     image_batch = torchvision.utils.make_grid(images, padding=0, nrow=4)
    #     imshow(image_batch)
    #
    #     break


import numpy as np
def compute_contrast(gray):

    # 计算图像的均值和标准差
    mean = np.mean(gray)
    std = np.std(gray)

    # 计算对比度值（使用标准差除以均值）
    contrast = std / mean

    return contrast

class LDM_test(LDM):
    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self.save_hyperparameters()
        self.init_from_ckpt(opts.ckpt_path)
        self.model_ema.copy_to(self.model)
        print("Switched to EMA weights")

    def test_step(self, batch, batch_idx):
        pathes = batch['path']
        pathes.extend(pathes)
        pathes.extend(pathes)


        #if os.path.exists(os.path.join(self.opts.result_save_dir, pathes[-1])): return

        mask_cond = batch['layer_cond'].to(self.device)
        class_cond = batch['class_cond'].to(self.device)

        mask_cond = torch.cat([mask_cond]*4,dim=0)
        class_cond = torch.cat([class_cond]*4,dim=0)

        mask_emb = self.model.mask_encoder(mask_cond)
        # print(mask_cond.shape)
        # print(mask_emb.shape)

        c = [mask_emb, class_cond]

        if self.opts.use_ddim:
            samples = self.ddim_sample(cond=c, batch_size=len(pathes))
        else:
            samples = self.sample(c=c, batch_size=len(pathes), return_intermediates=False, clip_denoised=True)

        samples = self.decode_first_stage(samples).cpu()
        samples = samples * 0.5 + 0.5
        #
        best_score = 0
        bi = 0
        save_dir = ''
        names = ''
        save_layer_dir = ''
        for i in range(len(pathes)):
            if '/' in pathes[i]:

                contrast = samples[i][0] * 255.
                score = compute_contrast(contrast.numpy())
                print(" image score:", score)

                names = pathes[i].split('/')[-5:]
                save_dir = '/'.join([self.opts.result_save_dir] + names[0:4])
                save_layer_dir = save_dir.replace('img','gt')
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(save_layer_dir,exist_ok=True)

                print(save_dir)
                if score > best_score:
                    best_score = score
                    bi = i
                # cv.imwrite(os.path.join(self.opts.result_save_dir, pathes[i]), samples[i])

        save_image(samples[bi:bi + 1], os.path.join(save_dir,names[-1]))
        save_image(mask_cond[bi:bi + 1], os.path.join(save_layer_dir, names[-1]))



    def ddim_sample(self, cond, batch_size, return_intermediates=False):
        ddim_sampler = DDIMSampler(self)
        shape = [self.channels] + list(self.latent_size)
        samples, intermediates = ddim_sampler.sample(self.opts.ddim_steps, batch_size,
                                                     shape, cond, verbose=False)
        if return_intermediates:
            return samples, intermediates
        else:
            return samples





if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    main(opts)
