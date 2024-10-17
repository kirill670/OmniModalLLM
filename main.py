# ==========================================
# Install Necessary Libraries
# ==========================================

# Uncomment and run these lines if you're setting up a new environment.
# They ensure the correct versions of PyTorch and other dependencies are installed.

# !pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# !pip install transformers datasets pillow fastapi uvicorn tiktoken einops tensorboard
# !pip install faiss-cpu slowapi

# ==========================================
# Imports and Device Initialization
# ==========================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import LongformerTokenizer, LongformerModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel
import io
from tqdm import tqdm
import uuid
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Attempt to import torch_xla for TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed import parallel_loader
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

def get_device():
    """Automatically select TPU, GPU, or CPU."""
    if TPU_AVAILABLE:
        device = xm.xla_device()
        print("Using device: TPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device

device = get_device()

# ==========================================
# Utility Functions
# ==========================================

def initialize_weights(module: nn.Module):
    """Initialize weights for linear and normalization layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """Load model checkpoint."""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint_data = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        epoch = checkpoint_data['epoch']
        loss = checkpoint_data['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {epoch}, loss {loss})")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return None, None

# ==========================================
# Regularization Modules
# ==========================================

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask

class DropBlock(nn.Module):
    """DropBlock regularization for spatial data."""
    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - (mask > 0).float()
        count = mask.numel() / mask.shape[0]
        return x * mask * (count / mask.sum())

class LayerDrop(nn.Module):
    """LayerDrop regularization for entire layers."""
    def __init__(self, drop_prob: float = 0.0):
        super(LayerDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return layer(x)
        if torch.rand(1).item() < self.drop_prob:
            return x
        return layer(x)

# ==========================================
# Liquid Layers
# ==========================================

class LiquidLinear(nn.Module):
    """Dynamic Linear layer with adaptive weights."""
    def __init__(self, in_features: int, out_features: int, adapt_dim: int):
        super(LiquidLinear, self).__init__()
        self.base_linear = nn.Linear(in_features, out_features)
        self.adapt_linear = nn.Linear(adapt_dim, out_features * in_features)
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        # Generate adaptive weights based on adapt_input
        adapt_weight = self.adapt_linear(adapt_input).view(self.base_linear.weight.size())
        # Combine base weights with adaptive weights
        weight = self.base_linear.weight + adapt_weight
        return F.linear(x, weight, self.base_linear.bias)

# ==========================================
# Vector Quantizer
# ==========================================

class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQVAE."""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # Flatten input
        flat_z = z.view(-1, self.embedding_dim)

        # Compute distances
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_z, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view(z.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

# ==========================================
# Variational Autoencoder (VAE) for Text
# ==========================================

class LiquidVAE(nn.Module):
    """VAE with LiquidLinear layers for text data."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, adapt_dim: int):
        super(LiquidVAE, self).__init__()
        self.encoder = nn.Sequential(
            LiquidLinear(input_dim, hidden_dim, adapt_dim),
            nn.GELU(),
            LiquidLinear(hidden_dim, hidden_dim, adapt_dim),
            nn.GELU()
        )
        self.fc_mu = LiquidLinear(hidden_dim, latent_dim, adapt_dim)
        self.fc_logvar = LiquidLinear(hidden_dim, latent_dim, adapt_dim)
        self.decoder = nn.Sequential(
            LiquidLinear(latent_dim, hidden_dim, adapt_dim),
            nn.GELU(),
            LiquidLinear(hidden_dim, input_dim, adapt_dim),
            nn.Sigmoid()
        )
        self.apply(initialize_weights)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        encoded = self.encoder(x, adapt_input)
        mu = self.fc_mu(encoded, adapt_input)
        logvar = self.fc_logvar(encoded, adapt_input)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, adapt_input)
        return {"reconstructed": reconstructed, "mu": mu, "logvar": logvar}

    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss: Reconstruction + KL Divergence."""
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# ==========================================
# Tokenizers
# ==========================================

class BaseTokenizer(nn.Module):
    """Base tokenizer class."""
    def __init__(self):
        super(BaseTokenizer, self).__init__()

    def tokenize(self, data: Any) -> torch.Tensor:
        raise NotImplementedError

    def detokenize(self, tokens: torch.Tensor) -> Any:
        raise NotImplementedError

class TextTokenizer(BaseTokenizer):
    """Tokenizer for text data using Longformer."""
    def __init__(self, encoder: LongformerTokenizer, adapt_dim: int):
        super(TextTokenizer, self).__init__()
        self.encoder = encoder
        self.vocab_size = self.encoder.vocab_size
        self.pad_token = self.encoder.pad_token_id
        self.embedding = nn.Embedding(self.vocab_size, 512)
        self.adapt_dim = adapt_dim
        self.apply(initialize_weights)

    def tokenize(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Tokenize input text and convert to embeddings."""
        tokens = self.encoder.encode(text, add_special_tokens=True)
        if len(tokens) < max_length:
            tokens += [self.pad_token] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # [1, max_length]
        embeddings = self.embedding(tokens_tensor)  # [1, max_length, embed_dim]
        return embeddings

    def detokenize(self, tokens: torch.Tensor) -> str:
        """Convert token embeddings back to text."""
        token_ids = tokens.argmax(dim=-1).cpu().numpy()[0]
        return self.encoder.decode(token_ids, skip_special_tokens=True)

class ImageTokenizer(BaseTokenizer):
    """Tokenizer for image data using VQVAE."""
    def __init__(self, device: str = 'cpu', num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super(ImageTokenizer, self).__init__()
        self.device = device
        self.vqvae = VQVAE(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost).to(self.device)
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def tokenize(self, image: Image.Image) -> torch.Tensor:
        """Tokenize image using VQVAE."""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)  # [1, 3, 128, 128]
        with torch.no_grad():
            quantized, _, _ = self.vqvae(image_tensor)  # [1, embedding_dim, 32, 32]
        tokens = quantized.permute(0, 2, 3, 1).contiguous().view(1, -1, quantized.shape[1])  # [1, 1024, embedding_dim]
        return tokens

    def detokenize(self, tokens: torch.Tensor) -> Image.Image:
        """Reconstruct image from tokens using VQVAE."""
        quantized = tokens.view(1, tokens.shape[-1], 32, 32)
        with torch.no_grad():
            reconstructed = self.vqvae.decoder(quantized)  # [1, 3, 128, 128]
        reconstructed = reconstructed.squeeze(0).cpu()
        reconstructed_image = transforms.ToPILImage()(reconstructed)
        return reconstructed_image

class LiquidFoundationTokenizer(nn.Module):
    """Foundation tokenizer handling both text and image modalities."""
    def __init__(self, device: str = 'cpu', adapt_dim: int = 256):
        super(LiquidFoundationTokenizer, self).__init__()
        self.encoder = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.text_tokenizer = TextTokenizer(self.encoder, adapt_dim=adapt_dim)
        self.image_tokenizer = ImageTokenizer(device=device)
        self.device = device

    def tokenize(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize input data containing text and/or image."""
        tokens = {}
        if 'text' in data and data['text'] is not None:
            tokens['text'] = self.text_tokenizer.tokenize(data['text'])
        if 'image' in data and data['image'] is not None:
            tokens['image'] = self.image_tokenizer.tokenize(data['image'])
        return tokens

    def detokenize(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Detokenize tokens back to original data."""
        data = {}
        if 'text' in tokens:
            data['text'] = self.text_tokenizer.detokenize(tokens['text'])
        if 'image' in tokens:
            data['image'] = self.image_tokenizer.detokenize(tokens['image'])
        return data

# ==========================================
# Mixture of Experts Components
# ==========================================

class KolmogorovArnoldExpert(nn.Module):
    """Kolmogorov-Arnold Expert with non-linear activations."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, activation: str = 'gelu'):
        super(KolmogorovArnoldExpert, self).__init__()
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Each phi function processes one input dimension
        self.phi_functions = nn.ModuleList([nn.Sequential(
            nn.Linear(1, hidden_dim),
            act_fn
        ) for _ in range(input_dim)])
        # Psi function combines all phi outputs
        self.psi_function = nn.Sequential(
            nn.Linear(input_dim * hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Kolmogorov-Arnold expert."""
        # Apply each phi function to corresponding input dimension
        phi_outputs = [phi(x[:, i].unsqueeze(1)) for i, phi in enumerate(self.phi_functions)]
        # Concatenate all phi outputs
        concatenated = torch.cat(phi_outputs, dim=1)
        # Apply psi function
        return self.psi_function(concatenated)

class MixtureOfExperts(nn.Module):
    """Mixture of Experts module with attention-based gating."""
    def __init__(
        self, 
        expert_dim: int, 
        num_experts: int, 
        adapt_dim: int, 
        hidden_dim: int = 64,
        drop_prob: float = 0.0,
        activation: str = 'gelu'
    ):
        super(MixtureOfExperts, self).__init__()
        # Create a list of LiquidLinear experts
        self.experts = nn.ModuleList([
            LiquidLinear(expert_dim, expert_dim, adapt_dim)
            for _ in range(num_experts)
        ])
        # Add Kolmogorov-Arnold expert
        self.ka_expert = KolmogorovArnoldExpert(expert_dim, expert_dim, hidden_dim, activation=activation)
        # Gating network to decide which expert to use
        self.gating = nn.Linear(adapt_dim, num_experts + 1)  # +1 for ka_expert
        self.drop_path = DropPath(drop_prob)
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mixture of Experts."""
        # Compute gating scores
        gate_scores = F.softmax(self.gating(adapt_input), dim=-1)  # [batch_size, num_experts + 1]
        expert_outputs = []
        # Compute outputs from each LiquidLinear expert
        for i, expert in enumerate(self.experts):
            expert_output = expert(x, adapt_input)  # [batch_size, expert_dim]
            expert_weight = gate_scores[:, i].unsqueeze(1)  # [batch_size, 1]
            expert_outputs.append(expert_weight * expert_output)
        # Compute output from Kolmogorov-Arnold expert
        ka_output = self.ka_expert(x)  # [batch_size, expert_dim]
        ka_weight = gate_scores[:, -1].unsqueeze(1)  # [batch_size, 1]
        expert_outputs.append(ka_weight * ka_output)
        # Sum all expert outputs
        output = sum(expert_outputs)  # [batch_size, expert_dim]
        # Apply DropPath regularization
        output = self.drop_path(output)
        return output

# ==========================================
# Component Combination
# ==========================================

class ComponentCombination(nn.Module):
    """Dynamically combines component outputs with learned weights."""
    def __init__(
        self, 
        input_dims: List[int], 
        hidden_dim: int = 128, 
        dropout_rate: float = 0.1, 
        activation: str = 'gelu',
        norm_type: str = 'batchnorm'
    ):
        super(ComponentCombination, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_components = len(input_dims)
        self.fc1 = nn.Linear(sum(input_dims), hidden_dim)
        
        # Choose activation function
        if activation == 'gelu':
            self.act1 = nn.GELU()
        elif activation == 'elu':
            self.act1 = nn.ELU()
        elif activation == 'leakyrelu':
            self.act1 = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Compute weights for each component
        self.fc2 = nn.Linear(hidden_dim, self.num_components)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.residual_fc = nn.Linear(sum(input_dims), sum(input_dims))
        
        # Choose normalization type
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(sum(input_dims))
        elif norm_type == 'groupnorm':
            self.norm = nn.GroupNorm(1, sum(input_dims))
        elif norm_type == 'instancenorm':
            self.norm = nn.InstanceNorm1d(sum(input_dims))
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        
        self.apply(initialize_weights)

    def forward(self, component_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass to combine component outputs."""
        # Check dimensions of each component output
        for i, (out, dim) in enumerate(zip(component_outputs, self.input_dims)):
            if out.shape[-1] != dim:
                raise ValueError(f"Component {i} dimension mismatch: expected {dim}, got {out.shape[-1]}")

        # Concatenate all component outputs
        concatenated = torch.cat(component_outputs, dim=-1)  # [batch, seq, sum(input_dims)]
        # Apply normalization
        x = concatenated.permute(0, 2, 1)  # [batch, sum(input_dims), seq]
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [batch, seq, sum(input_dims)]
        # Compute residual connection
        residual = self.residual_fc(concatenated)
        # Pass through fully connected layers
        x = self.fc1(concatenated)
        x = self.act1(x)
        x = self.dropout(x)
        # Compute weights for each component
        weights = self.fc2(x)  # [batch, seq, num_components]
        weights = self.softmax(weights)
        weights = weights.split(1, dim=-1)
        # Combine weighted component outputs
        combined_output = sum(w * out for w, out in zip(weights, component_outputs))
        # Add residual connection
        combined_output += residual
        return combined_output

# ==========================================
# Main LFModel with Gradient Checkpointing
# ==========================================

class LFModel(nn.Module):
    """Main model integrating LiquidLinear, MixtureOfExperts, attention, and component combination."""
    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        num_layers: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        max_drop_prob: float = 0.1,
        layerdrop_prob: float = 0.1,
        dropblock_block_size: int = 7,
        dropblock_prob: float = 0.1,
        combination_activation: str = 'gelu',
        combination_norm_type: str = 'batchnorm',
        norm_type: str = 'batchnorm',
        dynamic_layer_threshold: float = 0.5
    ):
        super(LFModel, self).__init__()
        # Featurizer to generate adaptive input
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        self.featurizer.apply(initialize_weights)

        # DropBlock regularization
        self.dropblock = DropBlock(block_size=dropblock_block_size, drop_prob=dropblock_prob)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            drop_prob = max_drop_prob * float(i) / float(num_layers)
            layer = nn.ModuleDict({
                'token_mixer': LiquidLinear(token_dim, token_dim, adapt_dim),
                'channel_mixer': LiquidLinear(channel_dim, channel_dim, adapt_dim),
                'moe': MixtureOfExperts(
                    expert_dim, num_experts, adapt_dim, hidden_dim=hidden_dim,
                    drop_prob=drop_prob,
                    activation='gelu'
                ),
                'attention': LongformerModel.from_pretrained('allenai/longformer-base-4096'),
                'combiner': ComponentCombination(
                    input_dims=[token_dim, channel_dim, expert_dim, 512],  # 512 for Longformer hidden size
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    activation=combination_activation,
                    norm_type=combination_norm_type
                ),
                'layerdrop': LayerDrop(layerdrop_prob)
            })
            self.layers.append(layer)
        
        self.dynamic_layer_threshold = dynamic_layer_threshold
        self.output_layer = nn.Linear(sum([token_dim, channel_dim, expert_dim, 512]), token_dim)
        self.output_layer.apply(initialize_weights)

    def forward(self, x: torch.Tensor, config_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Forward pass through the LFModel."""
        # Generate adaptive input by averaging over sequence dimension
        adapt_input = self.featurizer(x.mean(dim=1))  # [batch, adapt_dim]
        if config_weights is None:
            config_weights = {f"layer_{i+1}": 1.0 for i in range(len(self.layers))}
        for i, layer in enumerate(self.layers):
            layer_key = f"layer_{i+1}"
            layer_weight = config_weights.get(layer_key, 1.0)
            if layer_weight < self.dynamic_layer_threshold:
                continue
            # Apply LayerDrop with gradient checkpointing
            x = layer['layerdrop'](x, lambda x_inner: self._process_layer(layer, x_inner, adapt_input))
        # Apply DropBlock regularization
        x = self.dropblock(x)
        # Final output layer
        output = self.output_layer(x)
        return output

    def _process_layer(self, layer: nn.ModuleDict, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """Processes a single layer with gradient checkpointing."""
        def custom_forward(x_inner, adapt_input_inner):
            # Apply token mixer
            token_output = layer['token_mixer'](x_inner, adapt_input_inner)
            # Apply channel mixer
            channel_output = layer['channel_mixer'](x_inner, adapt_input_inner)
            # Apply Mixture of Experts
            moe_output = layer['moe'](x_inner, adapt_input_inner)
            # Apply Longformer attention
            attention_output = layer['attention'](x_inner)[0]  # [batch, seq, hidden]
            # Combine all component outputs
            component_outputs = [token_output, channel_output, moe_output, attention_output]
            combined_output = layer['combiner'](component_outputs)
            return combined_output
        return checkpoint(custom_forward, x, adapt_input)

# ==========================================
# Adaptive Configuration with Reflection Tuning
# ==========================================

class AdaptiveConfiguration(nn.Module):
    """Generates adaptive configuration weights with reflection tuning."""
    def __init__(self, adapt_dim: int):
        super(AdaptiveConfiguration, self).__init__()
        # Configuration network to generate initial weights
        self.config_net = nn.Sequential(
            nn.Linear(adapt_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4)
        )
        self.apply(initialize_weights)
        # Reflection network to adjust weights
        self.reflection_net = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, 4)
        )
        self.apply(initialize_weights)

    def forward(self, adapt_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to generate adaptive configuration weights."""
        # Generate initial configuration
        config = self.config_net(adapt_input)  # [batch, 4]
        config = F.softmax(config, dim=-1)
        # Apply reflection tuning
        reflection = self.reflection_net(config)
        reflection = torch.sigmoid(reflection)
        # Adjust configuration weights
        adjusted_config = config * reflection
        adjusted_config = F.softmax(adjusted_config, dim=-1)
        return {
            'moe_weight': adjusted_config[:, 0].unsqueeze(-1),
            'token_mixer_weight': adjusted_config[:, 1].unsqueeze(-1),
            'channel_mixer_weight': adjusted_config[:, 2].unsqueeze(-1),
            'attention_weight': adjusted_config[:, 3].unsqueeze(-1)
        }

# ==========================================
# Omnimodal LLM Integrating All Components
# ==========================================

class OmniModalLLM(nn.Module):
    """Omnimodal LLM handling text and image data with integrated token prediction."""
    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        num_layers: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        max_drop_prob: float = 0.1,
        layerdrop_prob: float = 0.1,
        dropblock_block_size: int = 7,
        dropblock_prob: float = 0.1,
        combination_activation: str = 'gelu',
        combination_norm_type: str = 'batchnorm',
        norm_type: str = 'batchnorm',
        dynamic_layer_threshold: float = 0.5
    ):
        super(OmniModalLLM, self).__init__()
        # Initialize LFModel
        self.lf_model = LFModel(
            token_dim=token_dim,
            channel_dim=channel_dim,
            expert_dim=expert_dim,
            adapt_dim=adapt_dim,
            num_experts=num_experts,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            max_drop_prob=max_drop_prob,
            layerdrop_prob=layerdrop_prob,
            dropblock_block_size=dropblock_block_size,
            dropblock_prob=dropblock_prob,
            combination_activation=combination_activation,
            combination_norm_type=combination_norm_type,
            norm_type=norm_type,
            dynamic_layer_threshold=dynamic_layer_threshold
        ).to(device)
        # Initialize LiquidVAE
        self.liquid_vae = LiquidVAE(input_dim=512, hidden_dim=256, latent_dim=128, adapt_dim=adapt_dim).to(device)
        # Initialize Adaptive Configuration
        self.adaptive_config = AdaptiveConfiguration(adapt_dim).to(device)
        # Token predictor to generate logits over vocabulary
        self.token_predictor = nn.Linear(512, 30522).to(device)  # Standard vocab size for Longformer
        self.token_predictor.apply(initialize_weights)

    def forward(self, text_embeddings: torch.Tensor, image_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the OmniModalLLM."""
        # Concatenate text and image embeddings along sequence dimension if image is provided
        if image_embeddings is not None:
            combined_input = torch.cat([text_embeddings, image_embeddings], dim=1)  # [batch, seq+img_seq, embed_dim]
        else:
            combined_input = text_embeddings  # [batch, seq, embed_dim]
        
        # Generate adaptive input by averaging over sequence dimension
        adapt_input = self.lf_model.featurizer(combined_input.mean(dim=1))  # [batch, adapt_dim]
        # Generate adaptive configuration weights
        config = self.adaptive_config(adapt_input)
        # Create config_weights dictionary for LFModel
        config_weights = {f"layer_{i+1}": weight.item() for i, weight in enumerate([
            config['moe_weight'], 
            config['token_mixer_weight'], 
            config['channel_mixer_weight'], 
            config['attention_weight']
        ])}
        # Pass through LFModel
        output = self.lf_model(combined_input, config_weights)  # [batch, total_seq, token_dim]
        # Split output into text and image parts if image is provided
        if image_embeddings is not None:
            seq_length = text_embeddings.shape[1]
            text_output = output[:, :seq_length, :]  # [batch, seq, token_dim]
        else:
            text_output = output  # [batch, seq, token_dim]
        # Compute mean of text output for VAE
        text_mean = text_output.mean(dim=1)  # [batch, token_dim]
        # Pass through LiquidVAE
        vae_outputs = self.liquid_vae(text_mean, adapt_input)
        reconstructed_text = vae_outputs["reconstructed"]  # [batch, token_dim]
        # Generate token logits
        token_logits = self.token_predictor(reconstructed_text)  # [batch, vocab_size]
        # Reconstruct full text using VAE decoder
        reconstructed_text_full = self.liquid_vae.decoder(vae_outputs["reconstructed"], adapt_input)  # [batch, input_dim]
        # Concatenate reconstructed text with image outputs if image is provided
        if image_embeddings is not None:
            combined_output = torch.cat([reconstructed_text_full.unsqueeze(1), output[:, seq_length:, :]], dim=1)  # [batch, total_seq, token_dim]
        else:
            combined_output = reconstructed_text_full  # [batch, input_dim]
        return {
            "output": combined_output,
            "token_logits": token_logits,
            "vae_reconstructed": vae_outputs["reconstructed"],
            "vae_mu": vae_outputs["mu"],
            "vae_logvar": vae_outputs["logvar"]
        }

    def save_model(self, path: str):
        """Save the model state."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the model state."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"Model loaded from {path}")

# ==========================================
# Vector Quantized VAE (VQVAE) Implementation
# ==========================================

class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder (VQVAE) for image tokenization."""
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32, 32]
            nn.ReLU(),
            nn.Conv2d(256, embedding_dim, kernel_size=3, stride=1, padding=1)  # [B, embedding_dim, 32, 32]
        )

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 128, 128]
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),  # [B, 3, 128, 128]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through VQVAE."""
        z_e = self.encoder(x)  # Encode input
        z_q, vq_loss, perplexity = self.vq_layer(z_e)  # Vector quantization
        x_recon = self.decoder(z_q)  # Reconstruct input
        return z_q, vq_loss, perplexity

# ==========================================
# Dataset Class for MS COCO
# ==========================================

class CocoDataset(Dataset):
    """Custom Dataset for MS COCO with text and image."""
    def __init__(self, dataset, tokenizer: TextTokenizer, image_transform):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.dataset[idx]
        text = sample['caption']
        image = sample['image']
        text_emb = self.tokenizer.tokenize(text)  # [1, max_length, embed_dim]
        image_emb = self.image_transform(image)  # [3, 128, 128]
        return {'text': text_emb.squeeze(0), 'image': image_emb}

# ==========================================
# Training Function
# ==========================================

def train_model(model, dataloader, optimizer, criterion, scheduler, device, num_epochs=1, save_path='checkpoint.pth.tar', patience=3):
    """Training loop with mixed precision and gradient checkpointing."""
    writer = SummaryWriter()
    scaler = GradScaler()
    best_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc="Training"):
            text_embeddings = batch['text'].to(device)  # [batch, seq, embed_dim]
            image_embeddings = batch['image'].to(device)  # [batch, 3, 128, 128]
            optimizer.zero_grad()
            with autocast():
                # Forward pass through the model
                outputs = model(text_embeddings, image_embeddings)
                token_logits = outputs["token_logits"]  # [batch, vocab_size]
                
                # Prepare labels: Assuming the labels are the next tokens in the sequence
                # Here, for simplicity, we're using the argmax of text embeddings as labels
                # In practice, you should have actual target token IDs
                labels = text_embeddings[:, 1:, :].argmax(dim=-1).reshape(-1)  # [batch*(seq-1)]
                token_logits = token_logits.reshape(-1, model.token_predictor.out_features)  # [batch*(seq-1), vocab_size]
                loss_tokens = criterion(token_logits, labels)
                
                # VAE loss
                vae_recon = outputs["vae_reconstructed"]  # [batch, token_dim]
                vae_mu = outputs["vae_mu"]
                vae_logvar = outputs["vae_logvar"]
                loss_vae = model.liquid_vae.loss_function(vae_recon, text_embeddings.mean(dim=1), vae_mu, vae_logvar)
                
                # Total loss
                loss = loss_tokens + loss_vae
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
            # Clear cache to free memory
            del loss, loss_tokens, loss_vae, outputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, filename=save_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
        
        # Step the scheduler
        scheduler.step(avg_loss)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    writer.close()

# ==========================================
# API Models
# ==========================================

class ChatMessage(BaseModel):
    role: str  # 'user', 'assistant', or 'system'
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # To manage conversation sessions
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    session_id: str
    message: ChatMessage

# ==========================================
# FastAPI Setup
# ==========================================

app = FastAPI()

# Initialize Limiter for Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==========================================
# Initialize Model and Tokenizer
# ==========================================

def initialize_model(device: str = 'cpu') -> (OmniModalLLM, LiquidFoundationTokenizer):
    """Initialize the OmniModalLLM and tokenizer with optimized settings."""
    token_dim = 512
    channel_dim = 512
    expert_dim = 256  # Reduced from 512 to save memory
    adapt_dim = 128    # Reduced from 256 to save memory
    num_experts = 4    # Reduced from 8 to save memory
    num_layers = 3     # Reduced from 6 to save memory
    hidden_dim = 64
    num_heads = 8

    tokenizer = LiquidFoundationTokenizer(device=device, adapt_dim=adapt_dim)
    model = OmniModalLLM(
        token_dim=token_dim,
        channel_dim=channel_dim,
        expert_dim=expert_dim,
        adapt_dim=adapt_dim,
        num_experts=num_experts,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=0.1,
        max_drop_prob=0.1,
        layerdrop_prob=0.1,
        dropblock_block_size=7,
        dropblock_prob=0.1,
        combination_activation='gelu',
        combination_norm_type='batchnorm',
        norm_type='batchnorm',
        dynamic_layer_threshold=0.5
    ).to(device)
    return model, tokenizer

# Initialize device as per earlier selection
model, tokenizer = initialize_model(device=device)
# Load pre-trained weights if available
# model.load_model('path_to_checkpoint.pth.tar')

# ==========================================
# Conversation History Storage
# ==========================================

# Simple in-memory storage for session histories
conversation_history = {}

# ==========================================
# Inference Function
# ==========================================

def generate_response(model: OmniModalLLM, tokenizer: LiquidFoundationTokenizer, user_text: str, session_id: str) -> str:
    """Generate assistant response based on user input and conversation history."""
    model.eval()
    with torch.no_grad():
        # Retrieve conversation history for the session
        history = conversation_history.get(session_id, [])
        
        # Concatenate all user and assistant messages
        conversation = ""
        for msg in history:
            if msg.role == 'user':
                conversation += f"User: {msg.content}\n"
            elif msg.role == 'assistant':
                conversation += f"Assistant: {msg.content}\n"
        
        # Append the latest user message
        conversation += f"User: {user_text}\nAssistant:"
        
        # Tokenize the conversation
        text_emb = tokenizer.text_tokenizer.tokenize(conversation).to(device)  # [1, seq, embed_dim]
        
        # Pass through the model
        outputs = model(text_emb, image_embeddings=None)  # Assuming text-only for chat
        token_logits = outputs["token_logits"]  # [1, vocab_size]
        
        # Generate the assistant's response token
        predicted_token_id = torch.argmax(token_logits, dim=-1)  # [1]
        
        # Detokenize to get the response text
        response_text = tokenizer.text_tokenizer.detokenize(predicted_token_id.unsqueeze(0))
        
    return response_text.strip()

# ==========================================
# API Endpoint
# ==========================================

@app.post("/chat/", response_model=ChatResponse)
@limiter.limit("20/minute")  # Limit to 20 requests per minute per IP
async def chat_endpoint(request: ChatRequest, req: Request):
    """API endpoint to generate response based on chat messages."""
    # Generate a new session_id if not provided
    if not request.session_id:
        session_id = str(uuid.uuid4())
        conversation_history[session_id] = []
    else:
        session_id = request.session_id
        if session_id not in conversation_history:
            conversation_history[session_id] = []
    
    # Append incoming messages to the conversation history
    conversation_history[session_id].extend(request.messages)
    
    # Extract the latest user message
    user_message = next((msg for msg in request.messages if msg.role == 'user'), None)
    if not user_message:
        return ChatResponse(
            session_id=session_id,
            message=ChatMessage(role="assistant", content="I didn't receive any user message.")
        )
    
    # Generate assistant's response
    assistant_reply = generate_response(
        model, 
        tokenizer, 
        user_message.content, 
        session_id=session_id
    )
    
    # Append assistant's reply to the conversation history
    conversation_history[session_id].append(
        ChatMessage(role="assistant", content=assistant_reply)
    )
    
    return ChatResponse(
        session_id=session_id,
        message=ChatMessage(role="assistant", content=assistant_reply)
    )

# ==========================================
# Main Function to Train and Launch API
# ==========================================

def main():
    """Main function to train the model and launch the API server."""
    # Data Loading and Preparation
    print("Loading MS COCO dataset...")
    dataset = load_dataset("coco_captions", "2017", split='train')
    # Filter out samples without captions or images
    dataset = dataset.filter(lambda x: len(x['caption']) > 0 and x['image'] is not None)
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    # Create custom dataset
    coco_dataset = CocoDataset(dataset, tokenizer.text_tokenizer, transform)
    # Determine batch size based on device
    if device.type == 'cuda':
        batch_size = 4  # Reduced from 8 to save memory
    elif device.type == 'xla':
        batch_size = 2  # Reduced for TPU
    else:
        batch_size = 2  # Reduced for CPU
    # Create DataLoader with adjusted batch size
    dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Optimizer, Loss, Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Training
    print("Starting training...")
    train_model(model, dataloader, optimizer, criterion, scheduler, device, num_epochs=5, save_path='checkpoint.pth.tar', patience=3)
    print("Training completed.")

    # Launch API with Uvicorn
    print("Launching API server...")
    # Run Uvicorn in a separate thread to allow training and API to run simultaneously
    import threading

    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    print("API server is running. You can send requests to /chat/.")

    # Keep the main thread alive to allow the API to run
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down.")

if __name__ == "__main__":
    main()
