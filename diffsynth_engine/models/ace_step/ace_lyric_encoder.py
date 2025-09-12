from typing import Tuple
import math
from einops import rearrange
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = nn.SiLU()
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(xs)))


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
    """

    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = n_feat // n_head
        self.q = nn.Linear(n_feat, n_feat)
        self.k = nn.Linear(n_feat, n_feat)
        self.v = nn.Linear(n_feat, n_feat)
        self.o = nn.Linear(n_feat, n_feat)
        self.p = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(n_head, self.d_k))

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).

        Returns:
            torch.Tensor: Output tensor.

        """
        b, h, t, d = x.shape
        zero_pad = torch.zeros((b, h, t, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)  # b h t (d+1)
        x_padded = x_padded.view(b, h, d + 1, t)
        # only keep the positions from 0 to time2
        x = x_padded[:, :, 1:].view_as(x)[..., : d // 2 + 1]
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            x (torch.Tensor): Query tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        p = self.p(pos_emb)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_head)
        k = rearrange(k, "b t (h d) -> b h d t", h=self.n_head)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_head)
        p = rearrange(p, "b t (h d) -> b h d t", h=self.n_head)

        q_with_bias_u = q + self.pos_bias_u[None, :, None, :]
        q_with_bias_v = q + self.pos_bias_v[None, :, None, :]
        matrix_ac = torch.matmul(q_with_bias_u, k)
        matrix_bd = torch.matmul(q_with_bias_v, p)
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        mask = mask.eq(0)[:, None, :, : scores.shape[-1]]
        scores = scores.masked_fill(mask, -float("inf"))
        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        x = torch.matmul(attn, v)
        x = rearrange(x, "b h t d -> b t (h d)")
        return self.o(x)


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
    """

    def __init__(
        self,
        size: int,
        num_heads: int,
        linear_units: int,
    ):
        super().__init__()
        self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        self.self_attn = RelPositionMultiHeadedAttention(num_heads, size)
        self.norm_ff = nn.LayerNorm(size)  # for the FNN module
        self.feed_forward = PositionwiseFeedForward(size, linear_units)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
        """
        x += self.self_attn(self.norm_mha(x), mask, pos_emb)
        x += self.feed_forward(self.norm_ff(x))
        return x, mask


class EspnetRelPositionalEncoding(nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1))
        return x, pos_emb

    def position_encoding(self, size: int) -> torch.Tensor:
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1 : self.pe.size(1) // 2 + size,
        ]
        return pos_emb


class LinearEmbed(nn.Module):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.

    """

    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
        )
        self.pos_enc = EspnetRelPositionalEncoding(odim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1024,
        attention_heads: int = 16,
        linear_units: int = 4096,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.embed = LinearEmbed(
            input_size,
            output_size,
        )
        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    attention_heads,
                    linear_units,
                )
                for _ in range(num_blocks)
            ]
        )
        self.after_norm = nn.LayerNorm(output_size)

    def forward(
        self,
        xs: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = pad_mask.to(torch.bool)[None]  # (B, 1, T)
        xs, pos_emb = self.embed(xs)
        for layer in self.encoders:
            xs, masks = layer(xs, masks, pos_emb)
        xs = self.after_norm(xs)
        return xs
