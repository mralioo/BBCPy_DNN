import torch.nn as nn

def xavier_initialize_weights(m):
    if isinstance(m, nn.Linear):  # Check if the module is a linear layer
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # Check if the layer has bias
            nn.init.zeros_(m.bias)