from enum import Enum


# FLUX ControlType
class ControlType(Enum):
    normal = "normal"
    bfl_control = "bfl_control"
    bfl_fill = "bfl_fill"
    bfl_kontext = "bfl_kontext"

    def get_in_channel(self):
        if self in [ControlType.normal, ControlType.bfl_kontext]:
            return 64
        elif self == ControlType.bfl_control:
            return 128
        elif self == ControlType.bfl_fill:
            return 384
