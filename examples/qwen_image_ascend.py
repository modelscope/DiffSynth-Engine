from diffsynth_engine import AttnImpl, QwenImagePipeline, QwenImagePipelineConfig, fetch_model


if __name__ == "__main__":
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="transformer/*.safetensors"),
        encoder_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="vae/*.safetensors"),
        device="npu:0",
        parallelism=1,
    )
    config.dit_attn_impl = AttnImpl.AUTO

    pipe = QwenImagePipeline.from_pretrained(config)
    image = pipe(
        prompt="A red panda reading a book beside a sunlit window",
        negative_prompt=" ",
        width=1328,
        height=1328,
        num_inference_steps=30,
        seed=42,
    )
    image.save("qwen_image_ascend.png")
