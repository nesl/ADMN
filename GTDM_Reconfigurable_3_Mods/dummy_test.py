import torch
def gumbel_sigmoid(
    logits, tau=1
):
    
    # ~Gumbel(0,1)`
    gumbels1 = (
        -torch.empty_like(logits)
        .exponential_()
        .log()
    )
    gumbels2 = (
        -torch.empty_like(logits)
        .exponential_()
        .log()
    )
    # Difference of two` gumbels because we apply a sigmoid
    gumbels1 = (logits + gumbels1 - gumbels2) / tau
    y_soft = gumbels1.sigmoid()
    return y_soft


  
import pdb; pdb.set_trace()