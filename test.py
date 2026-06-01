import os
import sys
from typing import Any

# Add project root so data.dataset_creator.image_utils can be imported
sys.path.insert(0, "/home/bingchen/image-model-studio")

import json, glob
import torch
from PIL import Image
from data.dataset_creator.image_utils import concat_image_list, put_two_id_images_into_one_top_bottom_image, stack_image_list

import torch
from PIL import Image
from data.dataset_creator.image_utils import put_two_id_images_into_one_top_bottom_image

#======================= model creating ===================================
from diffsynth_engine import QwenImagePipeline, QwenImagePipelineConfig, fetch_model
from tqdm import tqdm



    
"""
gsutil cp gs://uscentral1_ephemeral/multimodal/model_training/bingchen/qwen_image_srpo_pickscore_v1/step-400.safetensors /mnt/localdisk/bingchen/models/srpo_trial1_4h.safetensors

"""


model_path_root = "/data/bingchen/models/Qwen/Qwen-Image-Edit-2511/"
"""
mkdir -p /mnt/localdisk/bingchen/.cache/diffsynth/Qwen/
cp -r /data/bingchen/models/Qwen/Qwen-Image-Edit-2511 /mnt/localdisk/bingchen/.cache/diffsynth/Qwen/
"""
model_path_root = "/mnt/localdisk/bingchen/.cache/diffsynth/Qwen/Qwen-Image-Edit-2511/"
def get_ckpt_paths(root, name_pattern):
    return sorted(glob.glob(os.path.join(root, name_pattern)))

config = QwenImagePipelineConfig.basic_config(
        model_path=get_ckpt_paths(model_path_root, name_pattern="transformer/*.safetensors"),
        encoder_path=get_ckpt_paths(model_path_root, name_pattern="text_encoder/*.safetensors"),
        vae_path=get_ckpt_paths(model_path_root, name_pattern="vae/*.safetensors"),
        parallelism=1,
        use_zero_cond_t=True,
    )
pipeline = QwenImagePipeline.from_pretrained(config)
pipeline.vae_tiled = True
vae_tile_size = 128
pipeline.vae_tile_size = (vae_tile_size, vae_tile_size)
pipeline.vae_tile_stride = (96, 96)
#pipeline.enable_cpu_offload(offload_mode="cpu_offload", offload_to_disk=False)
pipeline.compile()

@torch.inference_mode()
def gen(edit_images, prompt, target_resolution, cfg=1, num_steps=8, seed=-1):
    if isinstance(cfg, list):
        cfg_list = cfg 
        cfg = 1
    else:
        cfg_list = None
    inputs = {
        "input_image": edit_images, 
        "prompt": prompt,
        #"prompt": f"Put characters in picture 1 and picture 2, into the same position and style as in picture 3. With picture 1's character on the left, picture 2's character on the right. And in the same artistic style of picture 3.",
        "cfg_scale": cfg,
        "cfg_list": cfg_list,
        "negative_prompt": "interwind arms, distortion, warped text, duplicate faces.",
        "num_inference_steps": num_steps,
        "seed": seed,
        "height": target_resolution[1],
        "width": target_resolution[0],
    }
    cur_result = pipeline(**inputs)
    return cur_result

infer_step = 8
step_lora_dict = {
    4: "/data/bingchen/models/Qwen/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    8: "/data/bingchen/models/Qwen/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
}

trained_lora_ckpts = [
    "/mnt/localdisk/bingchen/models/srpo_trial1_4h.safetensors",
]

pipeline.load_lora(path=trained_lora_ckpts[0], scale=1.0, fused=False)


#====================== prepare data =====================
data_root = "/data/bingchen/dataset/srpo_hpdv2/photo.json"
with open(data_root, "r") as f:
    prompt_list = json.load(f)
target_resolution = (768, 1360)
cfg = 1 #[2,2,1,1,1,1,1,1]
#target_resolution = (1440, 2560)
tester_nbr = 8

seed = 3333 #9999

output_root = "tmp_testing_outputs/srpo_test_v1"
os.makedirs(output_root, exist_ok=True)


pipeline.load_lora(path=trained_lora_ckpts[0], scale=1.0, fused=False)
new_col = []
for pi, cur_prompt in enumerate(prompt_list[:2]):

    res = gen(
            edit_images=None, 
            prompt=cur_prompt, 
            target_resolution=target_resolution, 
            cfg=4, 
            num_steps=40,
            seed=seed)
    new_col.append(res)

new_col = stack_image_list(new_col)
new_im_path = output_root+f'/tmp_qwen_{infer_step}_trained.jpg'
new_col.save(new_im_path)
        

"""
org compile:    2.78it/s
cuda 2 kernels: 2.79it/s
"""