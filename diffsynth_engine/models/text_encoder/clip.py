import torch
import torch.nn as nn

from diffsynth_engine.models.basic.attention import Attention


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        intermediate_size,
        num_heads=12,
        head_dim=64,
        use_quick_gelu=True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.attn = Attention(
            q_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            bias_q=True,
            bias_kv=True,
            bias_out=True,
            device=device,
            dtype=dtype,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim, device=device, dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(embed_dim, device=device, dtype=dtype)
        self.fc1 = nn.Linear(embed_dim, intermediate_size, device=device, dtype=dtype)
        self.fc2 = nn.Linear(intermediate_size, embed_dim, device=device, dtype=dtype)

        self.use_quick_gelu = use_quick_gelu

    def quickGELU(self, x):
        return x * torch.sigmoid(1.702 * x)

    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.use_quick_gelu:
            hidden_states = self.quickGELU(hidden_states)
        else:
            hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
