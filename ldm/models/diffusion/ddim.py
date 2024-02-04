"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer("ddim_alphas_next", np.append(ddim_alphas[1:],0.))
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        self.register_buffer("ddim_sqrt_recip_alphas", np.sqrt(1. / ddim_alphas))
        self.register_buffer("ddim_sqrt_recipm1_alphas", np.sqrt(1. / ddim_alphas - 1))

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               target_conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               org_mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               use_blend=False,
               percentage_of_pixel_blending=0.,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               use_reverse=False,
               use_deter_x=False,
               reverse_log_every_t=1,
               determinacy_x_T=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, list):
                for i, c in enumerate(conditioning):
                    if c.shape[0] != batch_size:
                        print(f"Warning: Got {i}th {c.shape[0]} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        if target_conditioning is None:
            target_conditioning = conditioning
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        if not use_reverse:
            samples, intermediates = self.ddim_sampling(conditioning, size,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, org_mask=org_mask, x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        use_deter_x=use_deter_x,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        use_blend=use_blend,
                                                        percentage_of_pixel_blending=percentage_of_pixel_blending,
                                                        )
        else:
            log_every_t = reverse_log_every_t
            samples, intermediates = self.ddim_reverse_sampling(conditioning, size,
                                                                callback=callback,
                                                                img_callback=img_callback,
                                                                quantize_denoised=quantize_x0,
                                                                x0=x0,
                                                                ddim_use_original_steps=False,
                                                                noise_dropout=noise_dropout,
                                                                temperature=temperature,
                                                                score_corrector=score_corrector,
                                                                corrector_kwargs=corrector_kwargs,
                                                                x_T=x_T,
                                                                log_every_t=log_every_t,
                                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                                unconditional_conditioning=unconditional_conditioning,
                                                                use_blend=use_blend,
                                                                percentage_of_pixel_blending=percentage_of_pixel_blending,
                                                                )

            # now we have x_0 x_t ... x_T images  ,  predict the mask
            log_every_t = 25
            intermediates_ddim_reverse = intermediates.copy()
            # """debug"""
            # XT_latent = intermediates['x_inter'][-1]
            # x1, x2, x3, x4 = XT_latent.split(1, 1)
            # x1 = x1[0]

            def custom_to_pil(x):
                x = x.detach().cpu()
                x = torch.clamp(x, -1., 1.)
                x = (x + 1.) / 2.
                x = x.permute(1, 2, 0).numpy()
                x = (255 * x).astype(np.uint8)
                from PIL import Image
                x = x.reshape(x.shape[0], x.shape[1])
                x = Image.fromarray(x)
                x = x.resize((256, 256))

                # if not x.mode == "RGB":
                #     x = x.convert("RGB")
                return x

            # custom_to_pil(x1).save("/home/xiebaoye/Documents/latent-diffusion/latent-diffusion-main/scripts/outputs/inpainting_results/"
            #                        "" + f"latent_ch_0.jpg")
            # print(len(XT_latent))
            """debug"""
            samples, intermediates = self.ddim_sampling(target_conditioning, size,
                                                        quantize_denoised=quantize_x0,
                                                        x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        use_deter_x=use_deter_x,
                                                        determinacy_x=intermediates['x_inter'],
                                                        determinacy_x_T=determinacy_x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        use_blend=use_blend,
                                                        percentage_of_pixel_blending=percentage_of_pixel_blending,
                                                        )
            # intermediates['x_inter'] = intermediates_ddim_reverse['x_inter']
        return samples, intermediates

    @torch.no_grad()
    def ddim_reverse_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      x0=None, img_callback=None, log_every_t=25,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, percentage_of_pixel_blending=0, use_blend=False):
        device = self.model.betas.device
        b = shape[0]
        eps = None

        img = x0

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [], 'pred_xT': []}
        #time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        time_range = range(0, timesteps) if ddim_use_original_steps else timesteps
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running reverse DDIM Sampling with {total_steps} timesteps")
        #time_range = reversed(time_range)
        iterator = tqdm(time_range, desc=' reverse DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = i

            # now_t = total_steps - 1 - i
            # k = time_range[now_t]

            ts = torch.full((b,), step, device=device, dtype=torch.long)

            #ccc = 1/0
            mean_pred, pred_xstart = self.p_reverse_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, eps=eps)
            img = mean_pred


            if index % log_every_t == 0 or index == total_steps - 1:
                img0 = mean_pred
                intermediates['x_inter'].append(mean_pred)
        intermediates['x_inter'].reverse()
        return img, intermediates # x_T


    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      use_deter_x=False,determinacy_x=None,org_mask=None,use_blend=False,
                      percentage_of_pixel_blending=0.,determinacy_x_T=None,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if use_deter_x == True and determinacy_x is not None and index > total_steps * 0.7:  # determinacy x_t  to prodict the mask_t
                img = determinacy_x[i]
                #img[:, 0:shape[1] - 1] = self.model.q_sample(x0, ts)[:, 0:shape[1] - 1]

                # img_orig = self.model.q_sample(x0, ts)
                #img[:, 0:shape[1] - 1] = img_orig[:, 0:shape[1] - 1]
                # va = 0.5  # 1. * i / len(time_range)
                # img[:, 0:shape[1]-1] = (1. - va) * determinacy_x[i] + va * img[:, shape[1]-1]
                # eps = 0.

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    def p_reverse_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, eps=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(
                2)  # classifier-free method e_t_uncond 无条件预测的噪声 / e_t 有条件预测的噪声
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # if score_corrector is not None:
        #     assert self.model.parameterization == "eps"
        #     e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_next = self.model.alphas_next_cumprod if use_original_steps else self.ddim_alphas_next
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        sqrt_recip_alphas = self.model.sqrt_recip_alphas_cumprod if use_original_steps else self.ddim_sqrt_recip_alphas
        sqrt_recipm1_alphas = self.model.sqrt_recipm1_alphas_cumprod if use_original_steps else self.ddim_sqrt_recipm1_alphas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        a_next_t = torch.full((b, 1, 1, 1), alphas_next[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        sqrt_recip_at = torch.full((b, 1, 1, 1), sqrt_recip_alphas[index], device=device)
        sqrt_recipm1_at = torch.full((b, 1, 1, 1), sqrt_recipm1_alphas[index], device=device)

        if self.model.parameterization == 'eps':
            pred_xstart = sqrt_recip_at * x - sqrt_recipm1_at * e_t
        elif self.model.parameterization == 'x0':
            pred_xstart = e_t

        x0_t =(x - e_t * ( 1 - sqrt_one_minus_at))
        #pred_xstart = pred_xstart.clamp(-1, 1)
        #
        eps = (sqrt_recip_at * x - pred_xstart) / sqrt_recipm1_at

        mean_pred = pred_xstart * torch.sqrt(a_next_t) + torch.sqrt(1. - a_next_t) * eps

        if quantize_denoised:
            pred_xstart, _, *_ = self.model.first_stage_model.quantize(pred_xstart)

        return mean_pred, pred_xstart

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            #e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            e_t = score_corrector.modify_score(e_t, x, t, c, corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        if self.model.parameterization == 'eps':
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        elif self.model.parameterization == 'x0':
            pred_x0 = e_t
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * ( ( x - pred_x0 * a_t.sqrt() ) / sqrt_one_minus_at )

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
