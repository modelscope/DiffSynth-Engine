from ..common.test_case import VideoTestCase
from diffsynth_engine.pipelines import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.download import fetch_model


class TestWanVideoTP(VideoTestCase):
    @classmethod
    def setUpClass(cls):
        config = WanModelConfig(
            model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors"),
            t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
            vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
        )
        cls.pipe = WanVideoPipeline.from_pretrained(config, parallelism=4, use_cfg_parallel=True)

    def test_txt2video(self):
        video = self.pipe(
            prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            num_frames=41,
            width=480,
            height=480,
        )
        self.save_video(video, "wan_tp_t2v.mp4")
        del self.pipe
