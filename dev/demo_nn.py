#!/bin/env python3
from collections import OrderedDict
from anemoi.datasets import open_dataset
from anemoi.utils.data_structures import NestedTrainingSample, TorchNestedAnemoiTensor
import torch
from torch import nn
from pathlib import Path


def get_dataset():
    HERE = Path(__file__).parent
    path = HERE / ".." / "src" / "anemoi" / "training" / "config" / "dataloader" / "observations.yaml"
    with path.open("r") as f:
        import yaml

        cfg = yaml.safe_load(f)
    cfg = cfg["dataset"]
    return open_dataset(cfg)

class DummyEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.encoders["era"] = nn.Linear(101, 64)
        self.encoders["metar"] = nn.Linear(14, 64)
        self.encoders["noaa-atms"] = nn.Linear(32, 64)

        self.mixer = nn.Linear(64, 64)

    def forward(self, x: TorchNestedAnemoiTensor) -> OrderedDict:
        y = OrderedDict()

        y = []
        for encoder_key, xt in zip(self.encoders, x):
            encoder = self.encoders[encoder_key]
            yt = encoder(xt.T)
            y.append(yt)

        y = TorchNestedAnemoiTensor(y)
        # return y
        return self.mixer(torch.cat(y.as_list()))


dummy_model = DummyEncoderModel()
dummy_model.train()

def get_data(i):
    ds = get_dataset()
    i_s = [i, i + 1, i + 2, i + 3]
    print()
    print(f"-> Using data for {ds.dates[i_s[0]]} to {ds.dates[i_s[-1]]}")
    return NestedTrainingSample([ds[_] for _ in i_s], name_to_index=ds.name_to_index)

i = 27
data = get_data(i)
assert len(data) == 4  # 4 states
assert len(data[0]) == 3 # era5 + 2 satellites
assert len(data[1]) == 3 # era5 + 2 satellites

x = data[0]
y_ref = data[1]
print(x.name_to_index)
print(x)

# x = OrderedDict()
# x["seviri"] = torch.from_numpy(data[0].squeeze(axis=1).T)
# x["metar"] = torch.from_numpy(data[1].squeeze(axis=1).T)
# x["noaa-atms"] = torch.from_numpy(data[2].squeeze(axis=1).T)
# x["era"] = torch.from_numpy(data[3].squeeze(axis=1).T)

y = dummy_model(x)

print(f"Model input shapes: {[list(x_in.shape) for x_in in x]}")
# print(f"Model input shapes: {[obs.upper() + ': ' + str(list(x_in.shape)) for (obs, x_in) in x.items()]}")
print(f"Model output shape: {list(y.shape)}")
assert y.shape == (sum([xt.shape[-1] for xt in x]), 64)
