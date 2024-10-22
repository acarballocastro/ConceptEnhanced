import os
import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext
from PIL import Image

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from pathlib import Path
from datasets.CUB_loader import get_CUB_dataloaders_singleattr
from utils import set_seed, generate_concept_matrix
from ddpm_conditional_emb import Diffusion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from train_CNN import CNN, ComplexCNN

config = SimpleNamespace(
    num_samples = 15, # per class!
    num_classes = 2,
    mask = [0, 51], # Concept indices to mask
    num_concepts = 85, #112,

    use_ema = True,
    save_images = True,

    epochs = 1,
    noise_steps = 2000,
    seed = 42,
    batch_size = 6,
    img_size = 64,

    diffusion_dir = '.',
    save_dir = '.',
    model_selected = 'E1_awa_C1_52', # name of the folder where the model is stored
    embedding_type = 'embpos', # 'embpos' or 'embposneg' or 'doubleemb'

    eval_dir = '.',
    root_path = '.',
    test_path = '.',

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_workers=10,
    lr = 3e-4
)

def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--num_samples', type=int, default=config.num_samples, help='number of samples from each class to generate')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--mask', type=int, default=config.mask, help='concepts to mask', nargs='+')
    parser.add_argument('--num_concepts', type=int, default=config.num_concepts, help='number of concepts')

    parser.add_argument('--use_ema', type=int, default=config.use_ema, help='use ema model or not')
    parser.add_argument('--save_images', type=str, default=config.save_images, help='save sampled images')

    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')

    parser.add_argument('--eval_dir', type=str, default=config.eval_dir, help='path to CNN model directory')
    parser.add_argument('--root_path', type=str, default=config.root_path, help='path to data directory')
    parser.add_argument('--test_path', type=str, default=config.test_path, help='path to test pkl')
    
    parser.add_argument('--diffusion_dir', type=str, default=config.diffusion_dir, help='path to diffusion model directory')
    parser.add_argument('--save_dir', type=str, default=config.save_dir, help='path to save dataset')
    parser.add_argument('--model_selected', type=str, default=config.model_selected, help='name of selected model')
    parser.add_argument('--embedding_type', type=str, default=config.embedding_type, help='embedding type')
    
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--num_workers', type=int, default=config.num_workers, help='number of workers') 
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


class ImageSampler:
    def __init__(self, config):
        self.config = config
        self.diffusion = Diffusion(
            noise_steps=config.noise_steps, 
            img_size=64, 
            num_concepts=config.num_concepts, 
            num_classes=config.num_classes,
            device=config.device, 
            mask=config.mask, 
            embedding_type=config.embedding_type
        )

    # load pretrained model
    def load_pretrained_model(self):
        ckpt = torch.load(os.path.join(self.config.diffusion_dir, self.config.model_selected, "ckpt_1499.pt"))
        ema_ckpt = torch.load(os.path.join(self.config.diffusion_dir, self.config.model_selected, "ema_ckpt_1499.pt"))
        self.diffusion.model.load_state_dict(ckpt)
        self.diffusion.ema_model.load_state_dict(ema_ckpt)

    # sample images
    def sample_images(self):
        # labels = torch.cat((torch.Tensor([1] * self.config.num_samples).long(), torch.Tensor([0] * self.config.num_samples).long())).to(self.config.device)
        # sampled_images = self.diffusion.sample(use_ema=self.config.use_ema, labels=labels)
        concept_matrix = generate_concept_matrix(torch.tensor([1 if i in config.mask else 0 for i in range(config.num_concepts)])).repeat_interleave(config.num_samples, 0).to(self.config.device)
        sampled_images = self.diffusion.sample(use_ema=self.config.use_ema, concept_matrix=concept_matrix, mask=config.mask)
        logging.info("Sampling complete!")
        return sampled_images

    # save images
    def save_images(self, sampled_images):
        if self.config.save_images:
            logging.info("Saving images")    
            os.makedirs(os.path.join(self.config.save_dir, self.config.model_selected), exist_ok=True)
            for idx, img in enumerate(sampled_images):
                img_np = img.permute(1, 2, 0).squeeze().cpu().numpy()  
                # Convert numpy array to PIL image
                pil_img = Image.fromarray(img_np)
                # Save image using PIL
                pil_img.save(os.path.join(self.config.save_dir, self.config.model_selected, f'image_{idx}.png'))
            logging.info("Saving complete!")  


if __name__ == "__main__":
    parse_args(config)

    # Sampling images
    sampler = ImageSampler(config)
    sampler.load_pretrained_model()
    sampled_images = sampler.sample_images()
    sampler.save_images(sampled_images)
