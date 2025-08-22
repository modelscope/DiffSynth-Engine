import tqdm
import torch
from typing import List
from collections import OrderedDict

from diffsynth_engine.utils.memory.linear_regression import LinearRegression, r2_score
from diffsynth_engine.utils.platform import empty_cache


def _profile_activation_memory(model, forward_kwargs):
    empty_cache()

    memory_before_inference = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(**forward_kwargs)
    activation_memory = torch.cuda.max_memory_reserved() - memory_before_inference
    return activation_memory / 1024 / 1024 / 1024


def _profile_weight_memory(model_cls, init_kwargs, device, dtype):
    memory_before_init = torch.cuda.memory_allocated(device)
    model = model_cls(**init_kwargs).to(device=device, dtype=dtype).eval()
    weight_memory = torch.cuda.memory_allocated(device) - memory_before_init
    return model, weight_memory / 1024 / 1024 / 1024


def _get_product(record: dict, name: str):
    product = 1
    for key, value in record.items():
        if key.startswith(f"input#{name}#"):
            product *= value
    return product


def _forward_kwargs_to_record(forward_kwargs, key_inputs):
    record = OrderedDict()
    for name, indices in key_inputs.items():
        if isinstance(indices, list):
            for dim_index in indices:
                record[f"input#{name}#{dim_index}"] = forward_kwargs[name].shape[dim_index]
        else:
            record[f"input#{name}#"] = forward_kwargs[name]
        product = _get_product(record, name)
        if indices is not None:  # If indices is None, do not use product, because the element is a scalar
            record[f"input#{name}#product"] = product
        record[f"input#{name}#product_square"] = product**2
    return record


def profile_model_memory(model_cls, init_kwargs, forward_kwargs_list, key_inputs, device, dtype):
    model, weight_memory = _profile_weight_memory(model_cls, init_kwargs, device, dtype)
    records = []

    for forward_kwargs in tqdm.tqdm(forward_kwargs_list):
        activation_memory = _profile_activation_memory(model, forward_kwargs)
        record = _forward_kwargs_to_record(forward_kwargs, key_inputs)
        record["weight_memory"] = weight_memory
        record["activation_memory"] = activation_memory
        record["total_memory"] = activation_memory + weight_memory
        records.append(record)
    return records


class MemoryPredictModel:
    def __init__(self, key_inputs=None):
        self.key_inputs = key_inputs
        self.model = LinearRegression()

    def predict(self, forward_kwargs):
        if self.model is None or self.key_inputs is None:
            raise ValueError("Model not initialized, please call from_pretrained first")
        input_args = _forward_kwargs_to_record(forward_kwargs, self.key_inputs)
        return self.model.predict(list(input_args.values()))[0].item()

    def train(self, records: List[dict]):
        if self.key_inputs is None:
            raise ValueError("Key inputs not set, please set key_inputs")
        X = []
        y = []
        for record in records:
            X.append(list({key: value for key, value in record.items() if key.startswith("input#")}.values()))
            y.append(record["total_memory"])
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            torch.save(
                {
                    "key_inputs": self.key_inputs,
                    "model": self.model.serialize(),
                },
                f,
            )

    @classmethod
    def from_pretrained(cls, model_path):
        with open(model_path, "rb") as f:
            data = torch.load(f)
            model = cls(data["key_inputs"])
            model.model = LinearRegression.deserialize(data["model"])
            return model
