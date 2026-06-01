"""Qwen Image LoRA Example

Demonstrates the full LoRA lifecycle using QwenImagePipeline:
  1. Load a LoRA model
  2. Generate with LoRA active
  3. Activate / deactivate LoRA and adjust scale dynamically
  4. Set selected LoRAs active
  5. Merge LoRA into base weights for faster inference
  6. Unmerge and reload LoRA
  7. Unload unmerged LoRA weights
"""

import os

import torch

from diffsynth_engine.pipelines.qwen_image import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model


def main():
    # 1. Initialize pipeline
    model_path = fetch_model("Qwen/Qwen-Image")
    pipe = QwenImagePipeline.from_pretrained(model_path_or_config=model_path)

    prompt = "A cat wearing a tiny samurai armor, digital painting"
    negative_prompt = "ugly, blurry, low quality"
    seed = 42

    def gen_kwargs():
        return dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=4.0,
            width=1024,
            height=1024,
            num_inference_steps=28,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )

    # 2. Generate without LoRA (baseline)
    image_base = pipe(**gen_kwargs()).images[0]
    image_base.save("qwen_image_lora_baseline.png")
    print("Saved baseline image.")

    # 3. Load distillation LoRA
    distill_dir = fetch_model("DiffSynth-Studio/Qwen-Image-Distill-LoRA")
    distill_path = os.path.join(distill_dir, "model.safetensors")
    pipe.load_loras(
        {
            "lora_id": "distill",
            "path": distill_path,
            "scale": 1.0,
        }
    )
    print("Distill LoRA loaded:", pipe.list_loras())

    # 4. Generate with distill LoRA (scale=1.0)
    image_distill = pipe(**gen_kwargs()).images[0]
    image_distill.save("qwen_image_lora_distill.png")
    print("Saved distill LoRA image (scale=1.0).")

    # 5. Adjust LoRA scale dynamically
    pipe.activate_loras("distill", scales=0.5)
    image_half = pipe(**gen_kwargs()).images[0]
    image_half.save("qwen_image_lora_distill_scale0.5.png")
    print("Saved distill LoRA image (scale=0.5).")

    # 6. Deactivate LoRA without unloading it
    pipe.deactivate_loras("distill")
    print("LoRA deactivated:", pipe.list_loras())

    image_deactivated = pipe(**gen_kwargs()).images[0]
    image_deactivated.save("qwen_image_lora_deactivated.png")
    print("Saved image with LoRA deactivated (should match baseline).")

    # 7. Reactivate LoRA
    pipe.activate_loras("distill", scales=1.0)
    print("LoRA activated:", pipe.list_loras())

    # 8. Merge LoRA into base weights (faster inference, no per-layer overhead)
    # high_precision=True (default) computes merge in float32 for better accuracy
    pipe.merge_loras(high_precision=True)
    print("LoRA merged:", pipe.list_loras())

    image_merged = pipe(**gen_kwargs()).images[0]
    image_merged.save("qwen_image_lora_merged.png")
    print("Saved image with merged LoRA (should match scale=1.0 result).")

    # 9. Unmerge and restore base model weights
    pipe.unmerge_loras()
    print("LoRA unmerged:", pipe.list_loras())

    # 10. Load multiple LoRAs (distill + aesthetic enhancement)
    art_aug_dir = fetch_model("DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1")
    art_aug_path = os.path.join(art_aug_dir, "model.safetensors")
    pipe.load_loras(
        [
            {"lora_id": "distill_v2", "path": distill_path, "scale": 0.8},
            {"lora_id": "art_aug", "path": art_aug_path, "scale": 0.8},
        ]
    )
    print("Multiple LoRAs loaded:", pipe.list_loras())

    pipe.set_active_loras("distill_v2", scales=0.8)
    print("Only distill_v2 set active:", pipe.list_loras())

    pipe.activate_loras("art_aug", scales=0.8)
    print("art_aug activated in addition to distill_v2:", pipe.list_loras())

    image_multi = pipe(**gen_kwargs()).images[0]
    image_multi.save("qwen_image_lora_multi.png")
    print("Saved image with distill + aesthetic LoRAs.")

    # 11. Unload specific LoRA
    pipe.unload_loras("art_aug")
    print("Unloaded art_aug:", pipe.list_loras())

    # 12. Reset all LoRA models
    pipe.reset_loras()
    print("All LoRA models reset:", pipe.list_loras())


if __name__ == "__main__":
    main()
