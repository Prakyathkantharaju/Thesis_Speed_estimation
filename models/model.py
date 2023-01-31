import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from positional_encodings.torch_encodings import PositionalEncodingPermute1D


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layer, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.conv1 = nn.Conv1d(6, d_model, 1)
        
         