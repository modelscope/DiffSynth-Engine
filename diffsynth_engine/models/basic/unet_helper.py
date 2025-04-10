import torch
import torch.nn as nn

from diffsynth_engine.models.basic.attention import Attention


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, device: str, dtype: torch.dtype):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, device=device, dtype=dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * nn.functional.gelu(gate)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim, num_attention_heads, attention_head_dim, cross_attention_dim, device: str, dtype: torch.dtype
    ):
        super().__init__()

        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, device=device, dtype=dtype)
        self.attn1 = Attention(
            q_dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias_out=True,
            device=device,
            dtype=dtype,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True, device=device, dtype=dtype)
        self.attn2 = Attention(
            q_dim=dim,
            kv_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias_out=True,
            device=device,
            dtype=dtype,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True, device=device, dtype=dtype)
        self.act_fn = GEGLU(dim, dim * 4, device=device, dtype=dtype)
        self.ff = nn.Linear(dim * 4, dim, device=device, dtype=dtype)

    def forward(self, hidden_states, encoder_hidden_states):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, y=encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.act_fn(norm_hidden_states)
        ff_output = self.ff(ff_output)
        hidden_states = ff_output + hidden_states

        return hidden_states


class DownSampler(nn.Module):
    def __init__(
        self, channels, padding=1, extra_padding=False, device: str = "cuda:0", dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding, device=device, dtype=dtype)
        self.extra_padding = extra_padding

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        if self.extra_padding:
            hidden_states = nn.functional.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class UpSampler(nn.Module):
    def __init__(self, channels, device: str, dtype: torch.dtype):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, device=device, dtype=dtype)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        hidden_states = nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels=None,
        groups=32,
        eps=1e-5,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True, device=device, dtype=dtype
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype
        )
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels, device=device, dtype=dtype)
        self.norm2 = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps, affine=True, device=device, dtype=dtype
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, device=device, dtype=dtype
            )

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None:
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)[:, :, None, None]
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


class AttentionBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        in_channels,
        num_layers=1,
        cross_attention_dim=None,
        norm_num_groups=32,
        eps=1e-5,
        need_proj_out=True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True, device=device, dtype=dtype
        )
        self.proj_in = nn.Linear(in_channels, inner_dim, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    device=device,
                    dtype=dtype,
                )
                for d in range(num_layers)
            ]
        )
        self.need_proj_out = need_proj_out
        if need_proj_out:
            self.proj_out = nn.Linear(inner_dim, in_channels, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states,
        time_emb,
        text_emb,
        res_stack,
        cross_frame_attention=False,
        **kwargs,
    ):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        if cross_frame_attention:
            hidden_states = hidden_states.reshape(1, batch * height * width, inner_dim)
            encoder_hidden_states = text_emb.mean(dim=0, keepdim=True)
        else:
            encoder_hidden_states = text_emb
            if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
                encoder_hidden_states = encoder_hidden_states.repeat(hidden_states.shape[0], 1, 1)

        for block_id, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if cross_frame_attention:
            hidden_states = hidden_states.reshape(batch, height * width, inner_dim)

        if self.need_proj_out:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = hidden_states + residual
        else:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        return hidden_states, time_emb, text_emb, res_stack


class PushBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        res_stack.append(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class PopBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        res_hidden_states = res_stack.pop()
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        return hidden_states, time_emb, text_emb, res_stack
