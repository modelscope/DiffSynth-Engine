import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inner_dim = dim * 4
        self.net = nn.ModuleList([GELU(dim, inner_dim), nn.Identity(), nn.Linear(inner_dim, dim, bias=True)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.weight = nn.Parameter(torch.empty((num_experts, embed_dim)))

    def linear(self, x):
        return F.linear(x, self.weight, None)

    def forward(self, hidden_states):
        logits = self.linear(rearrange(hidden_states, "b s d -> (b s) d"))
        return torch.topk(logits.softmax(dim=-1), k=self.top_k, dim=-1, sorted=False)


class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts=8, moe_top_k=2):
        super().__init__()
        self.moe_top_k = moe_top_k
        self.experts = nn.ModuleList([FeedForward(dim) for i in range(num_experts)])
        self.gate = MoEGate(embed_dim=dim, num_experts=num_experts, num_experts_per_tok=moe_top_k)
        self.shared_experts = FeedForward(dim)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_weight, topk_idx = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.moe_top_k
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum")
        return expert_cache
