import json

from diffsynth_engine.models.text_encoder.t5 import T5EncoderModel
from diffsynth_engine.utils.constants import ACE_TEXT_ENCODER_CONFIG_FILE


class ACETextEncoder(T5EncoderModel):
    @staticmethod # TODO: remove relative_position_embedding
    def get_model_config() -> dict:
        config_file = ACE_TEXT_ENCODER_CONFIG_FILE
        with open(config_file, "r") as f:
            config = json.load(f)
        return config