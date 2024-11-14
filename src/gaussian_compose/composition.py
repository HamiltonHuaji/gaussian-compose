import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import *

from .ops import indexed_transform

class Composition(nn.Module):
    def __init__(self, components: List[nn.ParameterDict], component_dims: Dict[str, Tuple[...]]):
        super().__init__()

        self.dims = component_dims
        assert "opacities" in self.dims, "opacities is required for composition"
        assert "means" in self.dims, "means is required for composition"
        assert "quats" in self.dims, "quats is required for composition"
        assert "scales" in self.dims, "scales is required for composition"

        self.attributes = nn.ParameterDict({
            key: nn.Parameter(torch.cat([
                component[key].view(-1, *self.dims[key])
                for component in components
            ]).detach())
            for key, dims in self.dims.items()
        })

        self.indices = torch.cat([
            torch.full_like(component["opacities"], i)
            for i, component in enumerate(components)
        ]).to(dtype=torch.int32)

        self.n = len(components)
        self.m = self.indices.size(0)

        assert self.indices.size() == (self.m, 1)
        assert torch.all(self.indices >= 0) and torch.all(self.indices < self.n)

        for key in self.dims:
            assert self.attributes[key].size() == (self.m, *self.dims[key])

    def transformed(self, trans: torch.Tensor, rotors: torch.Tensor, return_dict=False):
        assert trans.size() == (self.n, 3)
        assert rotors.size() == (self.n, 4)

        result_means, result_quats = indexed_transform(
            trans=trans,
            rotors=rotors,
            means=self.attributes["means"],
            quats=self.attributes["quats"],
            indices=self.indices,
            implementation="slang"
        )
        if return_dict:
            return {
                "means": result_means,
                "quats": result_quats,
            }
        else:
            return result_means, result_quats

def test():
    from .ops import _prepare_inputs
    components = [
        {
            "means": torch.randn(m, 3).cuda().requires_grad_(True),
            "quats": torch.randn(m, 4).cuda().requires_grad_(True),
            "scales": torch.randn(m, 3).cuda().requires_grad_(True),
            "opacities": torch.randn(m, 1).cuda().requires_grad_(True),
        }
        for m in [114514, 1919810]
    ]
    composition = Composition(components=components, component_dims={
        "means": (3,),
        "quats": (4,),
        "scales": (3,),
        "opacities": (1,),
    })

    trans = torch.randn(composition.n, 3).cuda().requires_grad_(True)
    rotors = torch.randn(composition.n, 4).cuda().requires_grad_(True)
    
    result_means, result_quats = composition.transformed(trans=trans, rotors=rotors, return_dict=False)
    print(result_means.shape)
    print(result_quats.shape)
