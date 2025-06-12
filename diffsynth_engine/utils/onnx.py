import onnxruntime
import diffsynth_engine.utils.logging as logging

logger = logging.get_logger(__name__)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class OnnxModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        if "cuda" in device:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        else:
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def forward(self, *args, **kwargs):
        inputs = {}
        for key, value in kwargs.items():
            inputs[key] = value
        for i, arg in enumerate(args):
            name = self.session.get_inputs()[i].name
            if name in inputs:
                raise ValueError(f"the input name [{name}] is duplicated")
            inputs[name] = arg
        return self.session.run(None, inputs)[0]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
