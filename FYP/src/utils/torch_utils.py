import torch.nn as nn


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)#std =0.02
        # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        # nn.init.kaiming_uniform_(module.in_proj_weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(module.out_proj.weight, nonlinearity='relu')
        module.in_proj_bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
