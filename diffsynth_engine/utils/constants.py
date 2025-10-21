import os

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)

# conf
CONF_PATH = os.path.join(PACKAGE_ROOT, "conf")

# tokenizers
FLUX_TOKENIZER_1_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "flux", "tokenizer_1")
FLUX_TOKENIZER_2_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "flux", "tokenizer_2")
SDXL_TOKENIZER_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "sdxl", "tokenizer")
SDXL_TOKENIZER_2_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "sdxl", "tokenizer_2")
WAN_TOKENIZER_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "wan", "umt5-xxl")
QWEN_IMAGE_TOKENIZER_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "qwen_image", "tokenizer")
QWEN_IMAGE_PROCESSOR_CONFIG_FILE = os.path.join(CONF_PATH, "tokenizers", "qwen_image", "qwen2_vl_image_processor.json")

# models
VAE_CONFIG_FILE = os.path.join(CONF_PATH, "models", "components", "vae.json")
FLUX_DIT_CONFIG_FILE = os.path.join(CONF_PATH, "models", "flux", "flux_dit.json")
FLUX_TEXT_ENCODER_CONFIG_FILE = os.path.join(CONF_PATH, "models", "flux", "flux_text_encoder.json")
FLUX_VAE_CONFIG_FILE = os.path.join(CONF_PATH, "models", "flux", "flux_vae.json")
SD_TEXT_ENCODER_CONFIG_FILE = os.path.join(CONF_PATH, "models", "sd", "sd_text_encoder.json")
SD_UNET_CONFIG_FILE = os.path.join(CONF_PATH, "models", "sd", "sd_unet.json")
SD3_DIT_CONFIG_FILE = os.path.join(CONF_PATH, "models", "sd3", "sd3_dit.json")
SD3_TEXT_ENCODER_CONFIG_FILE = os.path.join(CONF_PATH, "models", "sd3", "sd3_text_encoder.json")
SDXL_TEXT_ENCODER_CONFIG_FILE = os.path.join(CONF_PATH, "models", "sdxl", "sdxl_text_encoder.json")
SDXL_UNET_CONFIG_FILE = os.path.join(CONF_PATH, "models", "sdxl", "sdxl_unet.json")

WAN2_1_DIT_T2V_1_3B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.1_t2v_1.3b.json")
WAN2_1_DIT_T2V_14B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.1_t2v_14b.json")
WAN2_1_DIT_I2V_14B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.1_i2v_14b.json")
WAN2_1_DIT_FLF2V_14B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.1_flf2v_14b.json")
WAN2_2_DIT_TI2V_5B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.2_ti2v_5b.json")
WAN2_2_DIT_T2V_A14B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.2_t2v_a14b.json")
WAN2_2_DIT_I2V_A14B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.2_i2v_a14b.json")
WAN2_2_DIT_S2V_14B_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan2.2_s2v_14b.json")
WAN_DIT_KEYMAP_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "wan_dit_keymap.json")

WAN2_1_VAE_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "vae", "wan2.1_vae.json")
WAN2_2_VAE_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "vae", "wan2.2_vae.json")
WAN_VAE_KEYMAP_FILE = os.path.join(CONF_PATH, "models", "wan", "vae", "wan_vae_keymap.json")

QWEN_IMAGE_CONFIG_FILE = os.path.join(CONF_PATH, "models", "qwen_image", "qwen2_5_vl_config.json")
QWEN_IMAGE_VISION_CONFIG_FILE = os.path.join(CONF_PATH, "models", "qwen_image", "qwen2_5_vl_vision_config.json")
QWEN_IMAGE_VAE_CONFIG_FILE = os.path.join(CONF_PATH, "models", "qwen_image", "qwen_image_vae.json")
QWEN_IMAGE_VAE_KEYMAP_FILE = os.path.join(CONF_PATH, "models", "qwen_image", "qwen_image_vae_keymap.json")

# data size
KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
