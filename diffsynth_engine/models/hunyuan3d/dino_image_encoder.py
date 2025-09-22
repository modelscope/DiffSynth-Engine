import torch.nn as nn
import torchvision.transforms as transforms
import collections.abc
import math
from typing import Optional, Dict

import torch
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter

# --- 模块定义 (与原版基本一致，因为它们已经很明确了) ---


class Dinov2PatchEmbeddings(nn.Module):
    """
    将 (batch_size, num_channels, height, width) 的像素值转换为 (batch_size, seq_length, hidden_size) 的初始 patch 嵌入。
    """

    def __init__(self, image_size: int, patch_size: int, num_channels: int, hidden_size: int):
        super().__init__()
        image_size_tuple = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size_tuple = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        self.num_patches = (image_size_tuple[1] // patch_size_tuple[1]) * (image_size_tuple[0] // patch_size_tuple[0])
        self.image_size = image_size_tuple
        self.patch_size = patch_size_tuple
        self.num_channels = num_channels

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size_tuple, stride=patch_size_tuple)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                f"像素值的通道维度与配置中的通道维度不匹配。 预期 {self.num_channels} 但得到 {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class Dinov2Embeddings(nn.Module):
    """
    构造 CLS token、mask token、位置和 patch 嵌入。
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.patch_size = patch_size

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)

        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim).permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        pos_embed = self.interpolate_pos_encoding(embeddings, height, width)

        result = embeddings + pos_embed
        return result


class Dinov2SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} is not a multiple of num_attention_heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class Dinov2SelfOutput(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


# --- 重构开始：移除 **kwargs ---


class Dinov2Attention(nn.Module):
    # 重构：移除 **kwargs，明确所需参数
    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.attention = Dinov2SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
        )
        self.output = Dinov2SelfOutput(
            hidden_size=hidden_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs)
        return attention_output


class Dinov2LayerScale(nn.Module):
    def __init__(self, hidden_size: int, layerscale_value: float) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(layerscale_value * torch.ones(hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class Dinov2MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_features, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2Layer(nn.Module):
    """对应于原始实现中的 Block 类。"""

    # 重构：移除 **kwargs，明确所需参数
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: int,
        qkv_bias: bool,
        layer_norm_eps: float,
        layerscale_value: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = Dinov2Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
        )
        self.layer_scale1 = Dinov2LayerScale(hidden_size, layerscale_value)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = Dinov2MLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)
        self.layer_scale2 = Dinov2LayerScale(hidden_size, layerscale_value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.norm1(hidden_states))
        attention_output = self.layer_scale1(attention_output)
        hidden_states = attention_output + hidden_states

        layer_output = self.mlp(self.norm2(hidden_states))
        layer_output = self.layer_scale2(layer_output)
        layer_output = layer_output + hidden_states

        return layer_output


class Dinov2Encoder(nn.Module):
    # 重构：移除 **kwargs，明确所需参数
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        mlp_ratio: int,
        qkv_bias: bool,
        layer_norm_eps: float,
        layerscale_value: float,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                Dinov2Layer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    layer_norm_eps=layer_norm_eps,
                    layerscale_value=layerscale_value,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class Dinov2Model(nn.Module):
    """
    裸 DINOv2 模型，输出原始的 hidden-states。
    """

    # 重构：移除 **kwargs，明确所有模型参数
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        mlp_ratio: int,
        qkv_bias: bool,
        layer_norm_eps: float,
        layerscale_value: float,
    ):
        super().__init__()
        self.embeddings = Dinov2Embeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
        )
        self.encoder = Dinov2Encoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            layer_norm_eps=layer_norm_eps,
            layerscale_value=layerscale_value,
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        sequence_output = self.encoder(embedding_output)
        return self.layernorm(sequence_output)


class ImageEncoderStateDictConverter(StateDictConverter):
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("main_image_encoder."):
                new_key = key.replace("main_image_encoder.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict


class ImageEncoder(PreTrainedModel):
    converter = ImageEncoderStateDictConverter()

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__()
        # 定义模型超参数
        patch_size = 14

        # 定义实际输入图像的目标尺寸
        image_size = 518

        self.model = Dinov2Model(
            # Embedding 参数
            image_size=image_size,  # 模型权重预训练时的尺寸，用于位置编码
            patch_size=patch_size,
            num_channels=3,
            # Transformer 核心参数
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            mlp_ratio=4,
            # 细节参数
            qkv_bias=True,
            layer_norm_eps=1e-06,
            layerscale_value=1.0,
        )

        self.num_patches = (image_size // patch_size) ** 2 + 1  # 加上 cls token

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def forward(self, image, value_range=(-1, 1)):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        inputs = self.transform(image)
        outputs = self.model(inputs)
        return outputs
