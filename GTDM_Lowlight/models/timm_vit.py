
import torch
import torch.nn as nn

import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, layerdrop = 0.0, drop_layers = None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.layerdrop_rate = layerdrop
        self.drop_layers = drop_layers
        
        #self.dropout_block_enabled = 'block_fn' in kwargs.keys() and kwargs['block_fn'] == timm.models.vision_transformer.DropoutBlock

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
    # Forward train is when we are training the backbone
    # The entire BATCh has the same layerdrop configuration, dropped_layers is 12 length
    def forward_train(self, x, dropped_layers=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Allows us to specify which layers we want to drop at inference time
        drop_layers = []
        if self.drop_layers: 
            drop_layers = self.drop_layers
        # During training time, we pass in which layers we want to drop according to layerdrop ratio
        elif self.training and dropped_layers is not None:
            # 1 means we keep, 0 means we discard
            num_dropped = torch.sum(1 - dropped_layers) # Gives us how many layers we are dropping
            dropped_indices = torch.argsort(dropped_layers) # 0 -> num_dropped are the indices of dropped
            drop_layers = dropped_indices[0:num_dropped]
        # Go through each block, 
        for i, blk in enumerate(self.blocks):
            # Removed i == len(self.blocks) - 1
            if i not in drop_layers:
                x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    # No layerdrop for this, controller needs to run through first few layers
    def run_N_layers(self, x, num_layers):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for i in range(num_layers):
            x = self.blocks[i](x)
        return x
    
    # num_layers is where we stopped, controller_dropped_layers = b_size x total_layers, one hot: 0 for drop, 1 for not drop
    # Use this to enable batch training, we simply pass identify if we want to skip the layer
    # This is computationally ineffective, but we don't care too much about savings during training, during inference time we will not be batching
    def run_remaining_layers(self, x, num_layers, controller_dropped_layers):
        #if self.training:
        for i in range(num_layers, len(self.blocks)):
            if x.shape[0] == 1 and not self.training: # batch size is 1, testing
                if controller_dropped_layers[:, i]:
                    x = self.blocks[i](x)
            else:
                x_new = self.blocks[i](x)
                mask = controller_dropped_layers[:, i].cuda()
                x = x_new * mask[:, None, None] + (1 - mask[:, None, None]) * x
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome
    
    def forward_controller(self, x, controller_dropped_layers):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # if x.shape[0] == 1 and not self.training:
        #     for i, blk in enumerate(self.blocks):
        #         # batch size is 1, testing
        #             if controller_dropped_layers[0][i]:
        #                 x = blk(x)
        
        if x.shape[0] == 1 and not self.training:
            controller_dropped_layers = controller_dropped_layers[0].detach().cpu().tolist()
            for i, blk in enumerate(self.blocks):
                if controller_dropped_layers[i]:
                    x = blk(x)
        else:
            for i, blk in enumerate(self.blocks):
                x_new = blk(x)
                mask = controller_dropped_layers[:, i].cuda()
                x = x_new * mask[:, None, None] + (1 - mask[:, None, None]) * x
                # TODO explore impact of deatching 1 - mask
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    

