"""
Training of a conditional diffusion
Based on the code from https://github.com/tcapelle/Diffusion-Models-pytorch
"""

import os
import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

import wandb
from pathlib import Path
from utils import set_seed, one_batch, plot_images, save_images, generate_concept_matrix
from datasets.CUB_loader import get_CUB_dataloaders 
from datasets.AwA2_loader import get_AwA_dataloaders
from modules import UNet_conditional_embpos, UNet_conditional_doubleemb, UNet_conditional_embposneg, EMA

wandb.login() # login to wandb

config = SimpleNamespace(    
    run_name = "DDPM_conditional",
    epochs = 1500,
    noise_steps = 2000,
    seed = 42,
    batch_size = 5,
    img_size = 64,
    num_classes = 2, # how many concepts we will be looking at
    num_concepts = 85, # 112 for CUB
    mask = [7],
    data_path = '.',
    root_path = '.',
    train_path = '.',
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    slice_size = 1,
    do_validation = True,
    log_every_epoch = 25,
    num_workers=10,
    lr = 3e-4,
    embedding_type = "embpos")


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    """
    Diffusion model for image generation using UNet as the model architecture.
    Args:
        noise_steps: int, number of diffusion steps
        beta_start: float, starting value of beta
        beta_end: float, ending value of beta
        img_size: int, size of the input image
        num_classes: int, number of classes
        c_in: int, number of channels in the input image
        c_out: int, number of channels in the output image
        mask: list, concepts to mask
        embedding_type: str, type of embedding to use
        device: str, device to use for training
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, num_concepts=112, c_in=3, c_out=3, mask=[0], device="cuda", embedding_type="embpos", **kwargs):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        # initialize model
        if embedding_type == "embpos":
            self.model = UNet_conditional_embpos(c_in, c_out, num_concepts=num_concepts, **kwargs).to(device)
        elif embedding_type == "embposneg":
            self.model = UNet_conditional_embposneg(c_in, c_out, num_concepts=num_concepts, **kwargs).to(device)
        elif embedding_type == "doubleemb":
            self.model = UNet_conditional_doubleemb(c_in, c_out, num_concepts=num_concepts, **kwargs).to(device)
        else:
            raise ValueError("wrong embedding type")
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        
        self.c_in = c_in
        self.num_classes = num_classes
        self.num_concepts = num_concepts

        self.mask = torch.tensor([1 if i in mask else 0 for i in range(self.num_concepts)]).to(device)

    def prepare_noise_schedule(self):
        """
        Prepare the noise schedule for the diffusion process
        Returns:
            torch.Tensor: noise schedule
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        """
        Sample timesteps for diffusion process
        Args:
            n: int, number of timesteps to sample
        Returns:
            torch.Tensor: sampled timesteps
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        """
        Add noise to images at instant t
        Args:
            x: torch.Tensor, input images
            t: torch.Tensor, timesteps
        Returns:
            torch.Tensor: noise images
            torch.Tensor: noise
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, concept_matrix, mask, cfg_scale=3): #TODO: pass cfg_scale as parameter w
        """
        Sample new images from the model
        Args:
            use_ema:        bool, whether to use the EMA model
            concept_matrix: torch.Tensor, matrix of all potential combinations
            mask:           torch.Tensor, mask for concepts
            cfg_scale:      float, scale for conditional sampling
        Returns:
            torch.Tensor: sampled images
        """
        model = self.ema_model if use_ema else self.model
        n = concept_matrix.shape[0] 
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, concept_matrix, self.mask.repeat(n,1)) 
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2 # denormalization
        x = (x * 255).type(torch.uint8) # toTensor implicitly divides by 255
        return x

    def train_step(self, loss):
        """
        Perform a training step
        Args:
            loss: torch.Tensor, loss value
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        """
        Run one epoch of training or validation
        Args:
            train: bool, whether to run training or validation
        Returns:
            float: average loss value
        """
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, c) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                c = c.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1: # TODO: Probability for unconditional training - pass as parameter?
                    c = None
                predicted_noise = self.model(x_t, t, c, self.mask.repeat(images.shape[0],1))
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                wandb.log({"train_mse": loss.item(),
                            "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self):
        """
        Log images to wandb and save them to disk
        """
        concept_matrix = generate_concept_matrix(self.mask).to(self.device)
        sampled_images = self.sample(use_ema=False, concept_matrix=concept_matrix, mask=self.mask)
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, concept_matrix=concept_matrix, mask=self.mask)
        # plot_images(sampled_images)  #to display on jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        """
        Load model from checkpoint
        Args:
            model_cpkt_path: str, path to model checkpoint
            model_ckpt: str, model checkpoint name
            ema_model_ckpt: str, EMA model checkpoint name
        """
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        """
        Save model to disk and wandb
        Args:
            run_name: str, name of the run
            epoch: int, current epoch
        """
        torch.save(self.model.state_dict(), os.path.join(".", config.run_name, f"ckpt_{epoch}.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(".", config.run_name, f"ema_ckpt_{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(".", config.run_name, f"optim_{epoch}.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(".", config.run_name, f"scheduler_{epoch}.pt"))

    def prepare(self, args):
        """
        Prepare the model for training
        Args:
            args: dict, hyperparameters
        """
        # self.train_dataloader = get_CUB_dataloaders(args) 
        self.train_dataloader = get_AwA_dataloaders(
            classes_file=os.path.join(args.data_path, 'Animals_with_Attributes2/classes.txt'),
            data_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, 
            img_size=args.img_size, preload=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        """
        Train the model
        Args:
            args: dict, hyperparameters
        """
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})
            
            # log predicitons
            if epoch % args.log_every_epoch == 0:
                self.log_images()

            # save model checkpoint every 250 epochs
            if epoch % 250 == 0:
                self.save_model(run_name=args.run_name, epoch=epoch)
        
        # save model when training ends
        self.save_model(run_name=args.run_name, epoch=epoch)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--num_concepts', type=int, default=config.num_concepts, help='number of concepts')
    parser.add_argument('--root_path', type=str, default=config.root_path, help='root path to dataset')
    parser.add_argument('--train_path', type=str, default=config.train_path, help='path to train dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size') # 1/slice_size of the entire dataset
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--embedding_type', type=str, default=config.embedding_type, help='embedding type: embpos, embposneg, doubleemb')
    parser.add_argument('--mask', type=int, default=config.mask, help='concepts to mask', nargs='+')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, num_concepts=config.num_concepts, mask=config.mask, device=config.device, embedding_type=config.embedding_type)
    with wandb.init():
        path = Path(os.path.join(".", config.run_name)) # Path to load from
        path.mkdir(exist_ok=False)
        diffuser.prepare(config)
        diffuser.fit(config)
