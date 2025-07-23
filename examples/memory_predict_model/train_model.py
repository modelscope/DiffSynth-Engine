from diffsynth_engine.utils.memory.memory_predcit_model import MemoryPredictModel
import torch
import json


device = "cuda:0"
height = 4096
width = 4096

hidden_states = torch.randn(1, 16, height // 8, width // 8, device=device, dtype=torch.bfloat16)
timestep = torch.tensor([1000], device=device, dtype=torch.bfloat16)
prompt_emb = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)
pooled_prompt_emb = torch.randn(1, 768, device=device, dtype=torch.bfloat16)

guidance = torch.tensor([3.5], device=device, dtype=torch.bfloat16)
text_ids = torch.zeros(1, 512, 3, device=device, dtype=torch.bfloat16)

data = json.load(open("data.json"))

model = MemoryPredictModel(key_inputs={"hidden_states": [2, 3]})
r2 = model.train(data)
print(r2)
model.save_model("model.pth")
model = MemoryPredictModel.from_pretrained("model.pth")
result = model.predict(
    {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "prompt_emb": prompt_emb,
        "pooled_prompt_emb": pooled_prompt_emb,
        "guidance": guidance,
        "text_ids": text_ids,
        "image_emb": None,
    }
)

print(result)
