from types import SimpleNamespace
import torch 

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


def make_serializable(obj):
    if isinstance(obj, torch.nn.Parameter):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()

    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()

    elif isinstance(obj, SimpleNamespace):
        return {"__namespace__": {k: make_serializable(v) for k, v in vars(obj).items()}}

    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]

    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    else:
        return str(obj)