import argparse
from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
from skimage import transform
from scipy.ndimage import binary_erosion
from ldm.util import instantiate_from_config
from ldm.data.oct_datasets import *
from train_vqgan_uncond import  VQModel

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default=r'/xxx/xxx/last.ckpt')

    parser.add_argument('--data_config',
                        default=r'/configs/datasets.yaml')
    return parser

def main(opts):
    data_cfg = OmegaConf.load(opts.data_config)
    data_cfg.params.batch_size = 4
    data_cfg.params.train_data_tsv_paths = "/xxx/xxx/test_data.tsv"
    data = instantiate_from_config(data_cfg)
    model = VQModel_test(opts)
    data.setup()
    dataloader = data.train_dataloader()
    from tqdm import tqdm
    from matplotlib import pylab as plt
    import numpy as np
    for i, batch in enumerate(tqdm(dataloader)):
        xrec = model.test_step(batch,0)
        print(batch['path'])
        #z, _, _ = model.encode(batch['image'])
        #print(xrec.shape)
        def imshow(img):
            #img = img / 2 + 0.5  # unnormalize
            npimg = img.detach().numpy()
            npimg = (npimg + 1.0) / 2.0
            npimg = (npimg * 255).astype(np.uint8)
            plt.imshow(np.transpose(npimg, (1, 2, 0)),cmap='gray')  # 将【3，32，128】-->【32,128,3】
            plt.axis('off')
            plt.show()
        images = torch.cat([batch['image'], xrec], 0)
        image_batch = torchvision.utils.make_grid(images, padding=0, nrow=4)
        imshow(image_batch)
        break



class VQModel_test(VQModel):
    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self.init_from_ckpt(opts.ckpt_path)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = batch['image']
        pathes = batch['path']
        xrec, _, _ = self(x, return_pred_indices=True)
        return xrec

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=self.device)["state_dict"]
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



if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    main(opts)
