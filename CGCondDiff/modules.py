"""
Modules for the Concept-Guided Conditional Diffusion model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    """
    Exponential moving average of model parameters.
    Args:
        beta: Decay rate
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """ 
        Update the model parameters with the moving average of the parameters 
        Args:
            ma_model: Model with the moving average of the parameters
            current_model: Model with the current parameters  
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Update the average of the model parameters
        Args:
            old: Old model parameters
            new: New model parameters
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Update the model parameters with the moving average of the parameters
        Args:
            ema_model: Model with the moving average of the parameters
            model: Model with the current parameters
            step_start_ema: Number of steps to start the moving average
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Reset the parameters of the model with the moving average of the parameters
        Args:
            ema_model: Model with the moving average of the parameters
            model: Model with the current parameters
        """
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """ 
    Self-attention layer
    Args:
        channels: Number of channels
    """
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Forward pass of the self-attention layer
        Args:
            x: Input tensor
        Returns:
            attention_value: Output tensor
        """
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    """
    Double convolution layer
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Number of intermediate channels
        residual: Use residual connection
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        Forward pass of the double convolution layer
        Args:
            x: Input tensor
        Returns:
            x: Output tensor
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Downsample layer
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        emb_dim: Dimension of the embedding
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        """
        Forward pass of the downsample layer
        Args:
            x: Input tensor
            t: Time tensor
        Returns:
            x: Output tensor
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    """
    Upsample layer
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        emb_dim: Dimension of the embedding
    """
    def __init__(self, in_channels, out_channels, emb_dim=256): 
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        """
        Forward pass of the upsample layer
        Args:
            x: Input tensor
            skip_x: Skip tensor
            t: Time tensor
        Returns:
            x: Output tensor
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    """
    UNet model
    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        time_dim: Dimension of the time embedding
        remove_deep_conv: Remove deep convolutional layer
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        Positional encoding
        Args:
            t: Time tensor
            channels: Number of channels
        Returns:
            pos_enc: Positional encoding tensor
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        """
        Forward pass of the UNet model
        Args:
            x: Input tensor
            t: Time tensor
        Returns:
            output: Output tensor
        """
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        """
        Forward pass of the UNet model
        Args:
            x: Input tensor
            t: Time tensor 
        Returns:
            output: Output tensor
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    """
    Conditional UNet model
    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        time_dim: Dimension of the time embedding
        num_classes: Number of classes
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None): 
        """
        Forward pass of the conditional UNet model
        Args:
            x: Input tensor
            t: Time tensor
            y: Label tensor (concept)
        Returns:
            output: Output tensor
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)

class UNet_conditional_embpos(UNet):
    """
    Conditional UNet model (positive embedding)
    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        time_dim: Dimension of the time embedding
        c: Concept tensor
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_concepts=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)

        if num_concepts is not None:
            self.concept_emb = nn.Embedding(num_concepts, time_dim)

    def forward(self, x, t, c=None, m=None): 
        """
        Forward pass of the conditional UNet model
        Args:
            x: Input tensor
            t: Time tensor
            c: Concept tensor
            m: Mask tensor
        Returns:
            output: Output tensor
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if c is not None and m is not None:
            # Active concepts
            result = [torch.nonzero(idx) for idx in c&m]

            # Embedding of the active concepts
            embedding_averages = [self.concept_emb(idx).mean(dim=0) for idx in result]
            embedding_averages = torch.stack(embedding_averages).squeeze(dim=1)
            # In case no concept is active, we remove the resulting nan
            embedding_averages[embedding_averages.isnan()] = 0.0

            t += embedding_averages

        return self.unet_forwad(x, t)

class UNet_conditional_embposneg(UNet):
    """
    Conditional UNet model (opposite embedding)
    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        time_dim: Dimension of the time embedding
        c: Concept tensor
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_concepts=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)

        if num_concepts is not None:
            self.concept_emb = nn.Embedding(num_concepts, time_dim)

    def forward(self, x, t, c=None, m=None): 
        """
        Forward pass of the conditional UNet model
        Args:
            x: Input tensor
            t: Time tensor
            c: Concept tensor
            m: Mask tensor
        Returns:
            output: Output tensor
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if c is not None and m is not None:
            concept_mask = m * (2*c - 1)
            result = [torch.nonzero(idx) for idx in concept_mask]
            active_embedding = [self.concept_emb(idx) for idx in result]
            active_embedding = torch.stack(active_embedding).squeeze(dim=2)

            # Multiply rows by -1 for corresponding indices where c is -1
            embedding = torch.mul(active_embedding, concept_mask[:,(concept_mask != 0).any(dim=0)].unsqueeze(-1))

            # Mean of the embeddings
            mean_embedding = torch.mean(embedding, dim=1)

            t += mean_embedding

        return self.unet_forwad(x, t)

class UNet_conditional_doubleemb(UNet):
    """
    Conditional UNet model (double embedding)
    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        time_dim: Dimension of the time embedding
        c: Concept tensor
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_concepts=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)

        if num_concepts is not None:
            self.concept_emb_pos = nn.Embedding(num_concepts, time_dim)
            self.concept_emb_neg = nn.Embedding(num_concepts, time_dim)


    def forward(self, x, t, c=None, m=None): 
        """
        Forward pass of the conditional UNet model
        Args:
            x: Input tensor
            t: Time tensor
            c: Concept tensor
            m: Mask tensor
        Returns:
            output: Output tensor
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if c is not None and m is not None:
            # m = 1 and c = 1
            positive_mask = torch.logical_and(m == 1, c == 1).to(torch.int)
            result_pos = [torch.nonzero(idx) for idx in positive_mask]
            active_embedding_pos = [self.concept_emb_pos(idx) for idx in result_pos]

            # m = 1 and c = 0
            negative_mask = torch.logical_and(m == 1, c == 0).to(torch.int)
            result_neg = [torch.nonzero(idx) for idx in negative_mask]
            active_embedding_neg = [self.concept_emb_neg(idx) for idx in result_neg]

            embedding = torch.stack([torch.cat((active_embedding_pos[i], active_embedding_neg[i])) for i in range(t.shape[0])]).squeeze(dim=2)

            mean_embedding = torch.mean(embedding, dim=1)
            
            t += mean_embedding

        return self.unet_forwad(x, t)