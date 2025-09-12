import random

from diffsynth_engine.configs import ACEStepPipelineConfig
from diffsynth_engine.pipelines.ace_step import ACEStepMusicPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.audio import save_audio


if __name__ == "__main__":
    config = ACEStepPipelineConfig(
        model_path=fetch_model(
            model_uri="ACE-Step/ACE-Step-v1-3.5B",
            path="ace_step_transformer/diffusion_pytorch_model.safetensors",
        ),
    )
    seed = random.randint(0, 2**32 - 1)

    pipe = ACEStepMusicPipeline.from_pretrained(config)
    audio = pipe.text2audio(
        prompt="pop, rap, electronic, blues, hip-house, rhythm and blues",
        lyrics="[verse]\n我走过深夜的街道\n冷风吹乱思念的漂亮外套\n你的微笑像星光很炫耀\n照亮了我孤独的每分每秒\n\n[chorus]\n愿你是风吹过我的脸\n带我飞过最远最遥远的山间\n愿你是风轻触我的梦\n停在心头不再飘散无迹无踪\n\n[verse]\n一起在喧哗避开世俗的骚动\n独自在天台探望月色的朦胧\n你说爱像音乐带点重节奏\n一拍一跳让我忘了心的温度多空洞\n\n[bridge]\n唱起对你的想念不隐藏\n像诗又像画写满藏不了的渴望\n你的影子挥不掉像风的倔强\n追着你飞扬穿越云海一样泛光\n\n[chorus]\n愿你是风吹过我的手\n暖暖的触碰像春日细雨温柔\n愿你是风盘绕我的身\n深情万万重不会有一天走远走\n\n[verse]\n深夜的钢琴弹起动人的旋律\n低音鼓砸进心底的每一次呼吸\n要是能将爱化作歌声传递\n你是否会听见我心里的真心实意",
        audio_duration=170.63997916666668,
    )
    save_audio(audio, f"tmp/ace_t2m_{seed}")

    del pipe
