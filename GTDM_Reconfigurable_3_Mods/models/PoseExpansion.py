import torch
import torch.nn as nn

# All poses should go through the same network, used across all the modalities
class PoseExpand(nn.Module):

    def __init__(self, num_transformer_layers):
        # two layernorms per transformer encoder, 3 layers => 6 layernorms * 128 vit dim size * 2 params => 1536 params
        # To encourage the network to learn a good representation, maybe take an intermediate layer and concatenate to the adapter input?
        # Currently do one adaptive per layernorm
        super().__init__()
        self.out_dim = (2 * num_transformer_layers + 1) * 2
        self.intermediate_net = nn.Sequential(
            nn.Linear(7, 16),
            nn.GELU(), # TODO changed sigmoid back to GELU, does sigmoid hinder learning?
            nn.Linear(16, 16),
            # nn.GELU(),
            # nn.Linear(16, 32),
            # nn.GELU(),
            # nn.Linear(32, 32),
            # nn.GELU(),
            # nn.Linear(32, 32),
            # nn.Sigmoid(),
            # nn.Linear(32, 32)
        )
        self.output_net = nn.Sequential(
            nn.GELU(),
            nn.Linear(16, self.out_dim)
        )
        # self.intermediate_net = nn.Sequential(
        #     nn.Linear(7, 32),
        #     nn.GELU(), # TODO changed sigmoid back to GELU, does sigmoid hinder learning?
        #     nn.Linear(32, 32),
        #     nn.GELU(),
        #     nn.Linear(32, 64),
        #     nn.GELU(),
        #     nn.Linear(64, 64),
        #     nn.Sigmoid(),
        #     nn.Linear(64, 64)
        # )
        # self.output_net = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(64, self.out_dim)
        # )
    def forward(self, flattened_pose, get_intermediate=False):
        batch_num = flattened_pose.shape[0]
        intermediate = self.intermediate_net(flattened_pose)
        layernorm_vals = self.output_net(intermediate)
        # Num layernorm layers * 2 
        layernorm_vals = torch.reshape(layernorm_vals, (batch_num,-1, 2))
        return intermediate if get_intermediate else layernorm_vals
    