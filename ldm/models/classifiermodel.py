import os
from functools import partial
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
from taming.modules.vqvae.quantize import VectorQuantizer as VectorQuantizer
from ldm.modules.diffusionmodules.openaimodel import EncoderUNetModel
from collections import OrderedDict
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.util import log_txt_as_img, exists, default
from ldm.util import instantiate_from_config


class classififerModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 label_key="class_label",
                 scheduler_config=None,
                 learning_rate=1e-4,
                 lr_g_factor=1.0,
                 limit_step=750,
                 # init schedule
                 given_betas=None,
                 beta_schedule="linear",
                 timesteps=1000,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 ):
        super().__init__()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.instantiate_classifier_stage(ddconfig)
        #EncoderUNetModel(**ddconfig)


        self.loss_names = ['ce', 'acc']
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        self.image_key = image_key
        self.label_key = label_key

        self.loss = nn.CrossEntropyLoss()
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        # init schedule
        self.limit_step = limit_step
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        # vgqan encoder
        self.first_stage_model = None

        self.instantiate_first_stage(first_stage_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def instantiate_classifier_stage(self, config):
        model = instantiate_from_config(config)
        if model is None: return
        self.model = model

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        if model is None: return
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch, image_key, label_key):
        self.input = batch
        #print(type(batch[image_key]))
        #print(batch[image_key].shape)
        self.img = batch[image_key].transpose(1,3).transpose(2,3).to(self.device)


        self.gt_label = batch[label_key].to(self.device)

    def forward(self, batch, t=None, noise=None):
        self.get_input(batch, self.image_key, self.label_key)

        # x_t (random t)
        idx = 0
        if t is None:
            t = torch.randint(0, min(self.num_timesteps, self.limit_step), (self.img.shape[0],), device=self.device).long()
            idx = t.cpu().numpy()[0]

        x_start = self.img
        # latent x0
        if self.first_stage_model is not None:
            #print(f"before encoder x_start shape: {x_start.shape}")
            x_start = self.first_stage_model.encode(x_start)
        #print(f"x_start shape: {x_start.shape}")
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = x_start
        if idx > 0:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        self.pred_logit = self.model(x_noisy, timesteps=t)
        return self.pred_logit, x_noisy, idx

    def predict_latent_vector(self, image, t=None, noise=None):
        self.img = image
        x_start = self.img
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        self.pred_logit = self.model(x_noisy, timesteps=t)
        return self.pred_logit

    # def backward(self):
    #     self.loss_ce = self.criterion(self.pred_logit, self.gt_label)
    #     # print("loss: ",self.loss_ce)
    #     self.loss_ce.backward()
    #
    # def optimize_parameters(self, input):
    #     self.forward(input)
    #     self.optimizer.zero_grad()
    #     self.backward()
    #     self.optimizer.step()

    # def save(self, epoch):
    #     save_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.experiment_name,
    #                              f'epoch_{epoch}_net_')
    #     torch.save(self.model.state_dict(), save_path + self.model_name + '.pth')
    #     print(f'----save the {self.model_name} at epoch {epoch} successfully')

    # def load(self, epoch, order_path=None):
    #     if order_path is None:
    #         load_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.experiment_name,
    #                                  f'epoch_{epoch}_net_')
    #         state_dict = torch.load(load_path + self.model_name + '.pth', map_location=self.device)
    #         self.model.load_state_dict(state_dict)
    #         print(f'----load the pretrained {self.model_name} successfully----')
    #     else:
    #         self.model.load_state_dict(order_path)
    #         print(f'----load the pretrained {order_path.split("/")[-1]} successfully----')

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))

        return errors_ret

    @torch.no_grad()
    def evaluate(self):
        _, self.pred_label = torch.max(self.pred_logit.data, 1)
        self.correct = self.pred_label.eq(self.gt_label.long().data).sum()
        self.loss_acc = 100. * self.correct / np.shape(self.pred_label)[0]


    def on_train_batch_end(self, *args, **kwargs):
        return


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        pred_logit, _, idx = self(batch)

        self.loss_ce = self.criterion(self.pred_logit, self.gt_label)
        self.evaluate()
        log_dict_loss = self.get_current_losses()
        #self.log_dict(log_dict_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        loss_dict = {}
        loss = self.loss_ce * 50
        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_ce': loss.mean()})

        loss = self.loss_acc
        loss_dict.update({f'{log_prefix}/acc_loss': loss})
        loss_dict.update({f'{log_prefix}/xt_step': idx})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return self.loss_ce


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        self.get_input(batch, self.image_key, self.label_key)

        self.loss_ce = self.criterion(self.pred_logit, self.gt_label)

        self.evaluate()
        log_dict_loss = self.get_current_losses()

        self.log(f"val{suffix}/celoss", self.loss_ce,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict_loss)

        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.model.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae], scheduler
        return [opt_ae], []

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        pred_logit, x_t, idx = self(batch)
        if self.first_stage_model is not None:
            xrec = self.first_stage_model.decode(x_t)
        else:
            xrec = x_t
        x = self.img.to(self.device)
        self.evaluate()
        log["inputs"] = x
        log[f"x_{idx}_reconstruction_ac_loss_{self.loss_acc}"] = xrec
        return log


    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
