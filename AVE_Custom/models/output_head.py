# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class OutputHead(nn.Module):
    def __init__(self,
                 input_dim=256,
                 cov_add=1,
                 mlp_dropout_rate=0.0
                 ):
        super(OutputHead, self).__init__()
        self.include_z = False
        # Mean scale used to scale -1 to 1 to the actual cm values, a large mean scale pushes the sigmoid input towards 0, bad for learning
        self.mean_scale = [200, 200, 100] if self.include_z else [200, 200] #TODO Verify mean scale 
        # Two linear layers
        linear1 = nn.Linear(input_dim, input_dim)
        torch.nn.init.xavier_uniform_(linear1.weight)
        linear2 = nn.Linear(input_dim, input_dim)
        torch.nn.init.xavier_uniform_(linear2.weight)
        self.mlp = nn.Sequential(
            linear1,
            nn.GELU(), 
            linear1,
            nn.GELU(),
            nn.Dropout(mlp_dropout_rate)
        )

        # TODO Changed to 2D
        if self.include_z:
            self.num_outputs = 9
            self.mean_head = nn.Linear(input_dim, 3)
            self.cov_head = nn.Linear(input_dim, 6)
        else:
            self.num_outputs = 2 + 4
            self.mean_head = nn.Linear(input_dim, 2)
            self.cov_head = nn.Linear(input_dim, 4)
            
    
    
    #x has the shape B x num_object x D
    def forward(self, x):
        x = self.mlp(x)
        outputs = []
        result = {}
        # Run through mean and cov heads
        outputs.append(self.mean_head(x))
        outputs.append(self.cov_head(x))
        outputs = torch.cat(outputs, dim=-1)
        # No batch information
        if (len(outputs.shape) == 1):
            outputs = torch.unsqueeze(outputs, 0)
        if (self.include_z):
            # Mean is first 3, cov is the last 6, 3 diagonal and 3 off diagonal
            mean = outputs[..., 0:3]
            cov_logits = outputs[..., 3:9]
        else:
            mean = outputs[..., 0:2]
            cov_logits = outputs[..., 3:6]

        # Normalize mean to the range -1 to 1 so we can have negative values
        mean = (mean.sigmoid() - 0.5) * 2 # TODO tanh may be better
        #mean = torch.tanh(mean)
        mean = mean * torch.tensor(self.mean_scale).cuda() # Scale to the range of our outputs
        

        if (self.include_z):
            cov_diag = F.softplus(cov_logits[..., 0:3])
            cov_off_diag = cov_logits[..., 3:6]
            cov = torch.diag_embed(cov_diag)
            cov[..., 1, 0] = cov_off_diag[..., 0]
            cov[..., 2, 0] = cov_off_diag[..., 1]
            cov[..., 2, 1] = cov_off_diag[..., 2]
            cov = torch.bmm(cov, cov.transpose(-2,-1))
            cov = cov + torch.eye(3).cuda()
 
        else:
            cov_diag = F.softplus(cov_logits[..., 0:2])
            cov_off_diag = cov_logits[..., -1]
            cov = torch.diag_embed(cov_diag)    
            cov[..., -1, 0] += cov_off_diag
            cov = torch.bmm(cov, cov.transpose(-2,-1))
            cov = cov + torch.eye(2).cuda()


        
        # Place the results in the form of a MultivariateNormal distribution
        result['dist'] = []
        try:
            for batch_index in range(len(mean)):
                result['dist'].append(D.MultivariateNormal(mean[batch_index], cov[batch_index]))
            result['pred_mean'] = mean
            result['pred_cov'] = cov
        except:
            import pdb; pdb.set_trace()
        return result