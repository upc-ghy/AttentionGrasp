import torch.nn as nn


class PointModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)