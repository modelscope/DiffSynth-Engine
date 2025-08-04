from diffsynth_engine import fetch_model, QwenImagePipeline, QwenImagePipelineConfig


if __name__ == "__main__":
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="transformer/*.safetensors"),
        encoder_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="vae/*.safetensors"),
        parallelism=2,
    )
    pipe = QwenImagePipeline.from_pretrained(config)

    prompt = """
    这是一张用手机拍摄的广角照片，拍摄地点是一间俯瞰海湾大桥的房间，房间里有一块玻璃白板。视野中，一位女士正在写字，身穿一件印有 Qwen 大标志的 T 恤。字迹看起来很自然，略显凌乱，我们还能看到摄影师的倒影。

文字内容如下：

（左图）
“模态间的迁移：

假设我们直接用一个大型自回归变换器对
p(文本, 像素, 声音) [方程式] 进行建模。

优点：
* 利用丰富的世界知识增强图像生成
* 更高级别的文本渲染
* 原生上下文学习
* 统一的训练后堆栈

缺点：
* 不同模态的比特率不同
* 计算能力不自适应”

（右图）
“修复：
* 建模压缩表示
* 使用强大的解码器组合自回归先验”

在板子的右下角，她画了一张图：
“tokens -> [transformer] -> [diffusion] -> 像素”
    """
    negative_prompt = "ugly, blurry, low quality"
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=4.0,
        width=2560,
        height=1440,
        num_inference_steps=40,
        seed=42,
    )
    image.save("image.png")
    del pipe
