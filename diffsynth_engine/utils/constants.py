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

WAN_DIT_1_3B_T2V_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "1.3b-t2v.json")
WAN_DIT_14B_I2V_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "14b-i2v.json")
WAN_DIT_14B_T2V_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "14b-t2v.json")
WAN_DIT_14B_FLF2V_CONFIG_FILE = os.path.join(CONF_PATH, "models", "wan", "dit", "14b-flf2v.json")

# data size
KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB
