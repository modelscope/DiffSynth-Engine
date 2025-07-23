from diffsynth_engine.models.flux.flux_dit import FluxDiT
from diffsynth_engine.utils.memory.memory_predcit_model import profile_model_memory
import torch
import json

device = "cuda:0"

height = 1024
width = 1024

hidden_states = torch.randn(1, 16, height // 8, width // 8, device=device, dtype=torch.bfloat16)
timestep = torch.tensor([1000], device=device, dtype=torch.bfloat16)
prompt_emb = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)
pooled_prompt_emb = torch.randn(1, 768, device=device, dtype=torch.bfloat16)

guidance = torch.tensor([3.5], device=device, dtype=torch.bfloat16)
text_ids = torch.zeros(1, 512, 3, device=device, dtype=torch.bfloat16)

forward_kwargs_list = []
for i in range(1, 9):
    for j in range(1, 9):
        height = 256 * i
        width = 256 * j
        forward_kwargs_list.append(
            {
                "hidden_states": torch.randn(1, 16, height // 8, width // 8, device=device, dtype=torch.bfloat16),
                "timestep": timestep,
                "prompt_emb": prompt_emb,
                "pooled_prompt_emb": pooled_prompt_emb,
                "guidance": guidance,
                "text_ids": text_ids,
                "image_emb": None,
            }
        )

result = profile_model_memory(
    FluxDiT, {"in_channel": 64}, forward_kwargs_list, {"hidden_states": [2, 3]}, device, torch.bfloat16
)

json.dump(result, open("data.json", "w"))
