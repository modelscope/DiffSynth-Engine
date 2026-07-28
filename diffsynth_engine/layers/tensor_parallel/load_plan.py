import torch.nn as nn

from diffsynth_engine.layers.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from diffsynth_engine.layers.tensor_parallel.norm import TensorParallelRMSNorm
from diffsynth_engine.utils.load_utils import TensorSelection, TensorSelectionPlan, TensorSlice


def _slice_selection(dim: int, start: int, end: int) -> TensorSelection:
    return TensorSelection(slices=(TensorSlice(dim=dim, start=start, end=end),))


def derive_tp_model_weight_load_plan(model: nn.Module) -> TensorSelectionPlan:
    plan: TensorSelectionPlan = {}
    for name, module in model.named_modules():
        if isinstance(module, ColumnParallelLinear):
            start = module.tp_rank * module.out_features_per_partition
            end = start + module.out_features_per_partition
            plan[f"{name}.weight"] = _slice_selection(dim=0, start=start, end=end)
            if module.bias is not None:
                plan[f"{name}.bias"] = _slice_selection(dim=0, start=start, end=end)
        elif isinstance(module, RowParallelLinear):
            start = module.tp_rank * module.in_features_per_partition
            end = start + module.in_features_per_partition
            plan[f"{name}.weight"] = _slice_selection(dim=1, start=start, end=end)
        elif isinstance(module, TensorParallelRMSNorm) and module.weight is not None:
            start = module.tp_rank * module.hidden_size_per_partition
            end = start + module.hidden_size_per_partition
            plan[f"{name}.weight"] = _slice_selection(dim=0, start=start, end=end)
    return plan
