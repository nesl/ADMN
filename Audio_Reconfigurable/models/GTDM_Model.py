import numpy as np
import torch
import torch.nn as nn
# in-folder debugging
# import adapters as adapters
# import backbones as backbones
# import output_head as output_head
# from vit_dev import ViT, LIMUBertEnc, TransformerDec

import models.adapters as adapters
import models.output_head as output_head
from models.vit_dev import TransformerEnc, positionalencoding1d
from models.layer_controller import  ConvLayerController
from models.timm_vit import VisionTransformer
from timm.models.vision_transformer import Block 
from einops import rearrange
from timm.models.layers import to_2tuple



class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# This is the multimodal stage 1 model
# drop_layers_depth and drop_layers_img will be passed to it by the controller or by us during inference to test behavior with certain layers dropped
# vision_vit_layers and depth_vit_layers control how many layers we want to have at the start of training, if we set to 12 will load in MAE weights and freeze
# layerdrop specified the layerdrop rate we want to employ, active during training only
class GTDM_Early(nn.Module):
    def __init__(self, adapter_hidden_dim, valid_mods, drop_layers_img = None, drop_layers_audio=None, 
                 layerdrop=0.0, vision_vit_layers=12, audio_vit_layers=12):
        super(GTDM_Early, self).__init__()

        # Parameters used for multimodal fusion transformer
        dim_dec = 256
        depth_dec = 6 # Keep this for now but may need more to have good time understanding
        heads = 4

        if 'image' in valid_mods:
            # Initialize the vision backbone
            self.vision = VisionTransformer(
                patch_size=16, embed_dim=768, depth=vision_vit_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=nn.LayerNorm, layerdrop=layerdrop, drop_layers = drop_layers_img)
            
            # # Load in pretrained weights and freeze params if 12 layers ONLY
            # if vision_vit_layers == 12:
            #     print(self.vision.load_state_dict(torch.load('MAE_Dropout_FT_Dropout.pth')['model'], strict=False))
            #     # Freeze the parameters, leave only the last layer unfrozen
            #     for param in self.vision.parameters():
            #         param.requires_grad = False
            #     for param in self.vision.blocks[11].parameters():
            #         param.requires_grad = True
            
            # Transforms the output of the backbones into a common latent space
            self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)

        if 'audio' in valid_mods:
            # Initialize the depth transformer, keeping the last two layers unfrozen
            self.audio = VisionTransformer(
                patch_size=16, embed_dim=768, depth=audio_vit_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=nn.LayerNorm, layerdrop=layerdrop, drop_layers = drop_layers_audio)
            
            self.audio.patch_embed = PatchEmbed_new(img_size=(1024, 128), patch_size=(16,16), in_chans=1, embed_dim=768, stride=16) # no overlap. stride=img_size=16
            num_patches = self.audio.patch_embed.num_patches
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.audio.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False) 

            # # Load in pretrained weights and freeze params if 12 layers ONLY
            # if audio_vit_layers == 12:
            #     state_dict = torch.load('AudioMAE_Dropout_FT_Dropout.pth')['model']
            #     del state_dict['head.weight']
            #     del state_dict['head.bias']
            #     print(self.audio.load_state_dict(state_dict, strict=False))
            #     for param in self.audio.parameters():
            #         param.requires_grad = False
            #     for param in self.audio.blocks[11].parameters():
            #         param.requires_grad = True
        
            self.audio_adapter = adapters.Adapter(768, adapter_hidden_dim)

        # This performs multimodal fusion on the embeddings
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        
        # Outputs a multivariate normal distribution with mean and cov
        self.output_head = nn.Linear(dim_dec, 28)

        self.valid_mods = valid_mods

        
    # Returns an dictionary of object results with (modality, node) as the keys
    # At test time, we override the 0.2 layerdrop by passing in drop_layers_img and drop_layers_audio
    # in the constructor
    def forward(self, data):
        
        result_dict = {}
        outlist = []
        audio_data, img_data, _ = data # Audio data form b_size x 1 x 128 x 1024; Image data form b_size * 8 * 3 * 224 * 224, we stack 8 images
        if 'image' in self.valid_mods:
            # 1 indicates that we keep this layer, 0 indicates that it is dropped
            # During training, for a single batch and all the distributed sensors of one modality, use the same dropped layers, speeds up training
            if self.training: 
                dropped_layers_img = (torch.rand(12) > self.vision.layerdrop_rate).int().cuda()
                # Entire modality dropout
                if torch.rand(1).item() < 0.1:
                    dropped_layers_img = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).cuda()
            else:
                # If we are not training, do not drop any layers
                dropped_layers_img = torch.ones(12).int().cuda()
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w') # Compress stack dimension into batch
            out = self.vision.forward_train(img_data, dropped_layers_img) # Perform forward pass with dropped layers
            out = torch.squeeze(out)
            out = rearrange(out, '(b s) e -> b s e', b = audio_data.shape[0])
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            outlist.append(self.vision_adapter(out))
        # Same logic for audio
        if 'audio' in self.valid_mods:
            if self.training:
                dropped_layers_audio = (torch.rand(12) > self.audio.layerdrop_rate).int().cuda()
                if torch.rand(1).item() < 0.1:
                    dropped_layers_audio = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).cuda() # simulate entire modality dropout
            else:
                dropped_layers_audio = torch.ones(12).int().cuda()
        
            out = torch.squeeze(self.audio.forward_train(audio_data, dropped_layers_audio))
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            out = torch.unsqueeze(out, dim=1)
            outlist.append(self.audio_adapter(out))

        # Aggregate features together
        agg_features = torch.cat(outlist, dim=1)
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)
        # Perform multimodal fusion
        out = self.encoder(agg_features)
        out = torch.mean(out, dim=1) # Perform mean pooling
        # Get the class logits
        return self.output_head(out)


# Full Stage 2 Model with Controller
class Conv_GTDM_Controller(nn.Module):
    # Total layers establishes the layer budget that is provided to the controller
    def __init__(self, adapter_hidden_dim, valid_mods, total_layers=8):
        super(Conv_GTDM_Controller, self).__init__()

        depth_dec = 6
        heads = 4
        dim_dec = 256

        # In this version, we do not use the dynamic layerdrop or set the drop layers, this is all done by the model itself

        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, block_fn = Block)
 
        self.audio = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, block_fn = Block)
        
        self.audio.patch_embed = PatchEmbed_new(img_size=(1024, 128), patch_size=(16,16), in_chans=1, embed_dim=768, stride=16) # no overlap. stride=img_size=16
        num_patches = self.audio.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        self.audio.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False) 

        self.controller = ConvLayerController(total_layers=total_layers)
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        
        self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.audio_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.output_head = nn.Linear(dim_dec, 28)
        self.valid_mods = valid_mods


    # Controller temperature is the gumbel softmax temperature, discretization method describes if we are doing admn, only straight through, etc
    def forward(self, data, controller_temperature=1, discretization_method='admn'):
        outlist = []
        # Controller outputs the layers that should be dropped and the predicted noise of each modality
        dropped_layers, predicted_noise = self.controller(data, self.valid_mods, controller_temperature, discretization_method)
        audio_data, img_data, labels = data
        # Dropped_layers[:, 0] corresponds to image modality
        if 'image' in self.valid_mods:
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w') # Compress stack dimension into batch
            img_dropped_layers = dropped_layers[:, 0]
            # We have 8 images per sample that we expand in to the batch dimension
            img_dropped_layers = torch.repeat_interleave(img_dropped_layers, 8, dim=0)
            out = self.vision.forward_controller(img_data, img_dropped_layers) # Perform forward pass with dropped layers
            out = torch.squeeze(out)
            out = rearrange(out, '(b s) e -> b s e', b = audio_data.shape[0])
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            outlist.append(self.vision_adapter(out))
        # Dropped_layers[:, 1] is the audio modality dropped layers
        if 'audio' in self.valid_mods:
            out = torch.squeeze(self.audio.forward_controller(audio_data, dropped_layers[:, 1]))
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            out = torch.unsqueeze(out, dim=1)
            outlist.append(self.audio_adapter(out))

        agg_features = torch.cat(outlist, dim=1)
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)

        out = self.encoder(agg_features) #bs x total_patches x 256
        out = torch.mean(out, dim=1)
        
        return self.output_head(out), predicted_noise





# USED TO ONLY MEASURE FLOPS BY RETURNING TENSOR
class GTDM_Early_Test_FLOPS(nn.Module):
    def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes, drop_layers_img = None, drop_layers_depth=None, layerdrop=0.0, vision_vit_layers=12, depth_vit_layers=12):
        super(GTDM_Early_Test_FLOPS, self).__init__()
        # Parameters used for multimodal fusion transformer
        dim_dec = 256
        depth_dec = 6 # changed from 6 to 9
        heads = 4

        # Initialize the vision backbone
        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=vision_vit_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, layerdrop=layerdrop, drop_layers = drop_layers_img)
        if vision_vit_layers == 12:
            print(self.vision.load_state_dict(torch.load('MAE_Dropout_FT_Dropout.pth')['model'], strict=False))
            # Freeze the parameters, leave only the last layer unfrozen
            for param in self.vision.parameters():
                param.requires_grad = False
            for param in self.vision.blocks[11].parameters():
                param.requires_grad = True
        # Initialize the depth transformer, keeping the last two layers unfrozen
        self.depth = VisionTransformer(
            patch_size=16, embed_dim=768, depth=depth_vit_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, layerdrop=layerdrop, drop_layers = drop_layers_depth)
        #self.num_layer_embeds = nn.Embedding(13, dim_dec)
        if depth_vit_layers == 12:
            print(self.depth.load_state_dict(torch.load('MAE_Dropout_FT_Dropout.pth')['model'], strict=False))
            for param in self.depth.parameters():
                param.requires_grad = False
            for param in self.depth.blocks[10].parameters():
                param.requires_grad = True
            for param in self.depth.blocks[11].parameters():
                param.requires_grad = True

        # Use encoder to combine the information
        #self.fusion_cls = nn.Parameter(torch.randn(1, dim_dec))
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.depth_adapter = adapters.Adapter(768, adapter_hidden_dim)
 
        self.output_head = output_head.OutputHead()

        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes

        
    # Returns an dictionary of object results with (modality, node) as the keys
    # At test time, we override the 0.2 layerdrop by passing in drop_layers_img and drop_layers_depth
    # in the constructor
    def forward(self, data):
        
        result_dict = {}
        outlist = []
        b_size = len(data[('mocap', 'mocap')]['gt_positions'])
        if 'zed_camera_left' in self.valid_mods:
            # 1 indicates that we keep this layer, 0 indicates that it is dropped
            if self.training:
                # Initialize ONE noise vector for the entire batch and for all distributed cameras
                # This is bc we speed up training by avoiding computing zeroed out info
                dropped_layers_img = (torch.rand(12) > self.vision.layerdrop_rate).int().cuda()
                if torch.rand(1).item() < 0.1:
                    dropped_layers_img = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).cuda() # simulate entire modality dropout
            else:
                # If we are not training, do not drop any layers
                dropped_layers_img = torch.ones(12).int().cuda()
            for node in self.valid_nodes:
                node = str(node)
                data_transformed = data[('zed_camera_left', 'node_' + str(node))]
                out = self.vision.forward_train(data_transformed, dropped_layers_img)
                out = torch.squeeze(out)
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.vision_adapter(out))
        if 'realsense_camera_depth' in self.valid_mods:
            if self.training:
                dropped_layers_depth = (torch.rand(12) > self.depth.layerdrop_rate).int().cuda()
                if torch.rand(1).item() < 0.1:
                    dropped_layers_depth = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).cuda() # simulate entire modality dropout
            else:
                dropped_layers_depth = torch.ones(12).int().cuda()
            for node in self.valid_nodes:
                node = str(node)
                depth_data = data[('realsense_camera_depth', 'node_' + str(node))]
                out = torch.squeeze(self.depth.forward_train(depth_data, dropped_layers_depth))
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.depth_adapter(out))
        # Get all the dropped_layers and repeat along batch to get b_size x 2 x 12
        #dropped_layers_total = torch.stack((dropped_layers_img, dropped_layers_depth), dim=0)
        #dropped_layers_total = torch.stack([dropped_layers_total for _ in range(b_size)], dim=0)
        # Get number that the model still has (number of 1s), b_size x 2
        #num_present_layers = torch.sum(dropped_layers_total, dim=-1)
        
        # Convert to layer embeddings
        #layer_embeds = self.num_layer_embeds(num_present_layers)
        #cls_tokens = self.fusion_cls.expand(b_size, -1, -1)
        # Append the layer embeddings to the start, informing model about number of layers producing
        # each modality embedding
        agg_features = torch.stack(outlist, dim=1)
        #agg_features = torch.cat((cls_tokens, agg_features), dim=1)
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)

        out = self.encoder(agg_features)
        out = torch.mean(out, dim=1) # Perform mean pooling
        
        result_dict["early_fusion"] = self.output_head(out)['pred_mean'] # output_head returns a result of which we take the 'dist' key
        return result_dict
    



# USED TO ONLY MEASURE FLOPS BY RETURNING TENSOR
class Conv_GTDM_Controller_Test_FLOPS(nn.Module):
    # Total layers dictates how many cumulative layers we want to limit our model to
    def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes, total_layers=8):
        super(Conv_GTDM_Controller_Test_FLOPS, self).__init__()

        dim_dec = 256
        depth_dec = 6
        heads = 4

        # In this version, we do not use the dynamic layerdrop or set the drop layers, this is all done by the model itself

        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, block_fn = Block)
 
        self.depth = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, block_fn = Block)

        self.controller = ConvLayerController(total_layers=total_layers)
        #self.num_layers_embeds = nn.Embedding(12, dim_dec)
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        
        self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.depth_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.output_head = output_head.OutputHead()
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes


       
    # Returns an dictionary of object results with (modality, node) as the keys
    def forward(self, data, controller_temperature=5):
        
        result_dict = {}
        outlist = []
        dropped_layers, predicted_noise = self.controller(data, self.valid_mods, self.valid_nodes, controller_temperature)
        # 64 x n_mods x 12
        if 'zed_camera_left' in self.valid_mods:
            for node in self.valid_nodes: 
                vision_data = data[('zed_camera_left', 'node_' + str(node))]
                out = torch.squeeze(self.vision.forward_controller(vision_data, dropped_layers[:, 0]))
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.vision_adapter(out))
        if 'realsense_camera_depth' in self.valid_mods:
            for node in self.valid_nodes:
                depth_data = data[('realsense_camera_depth', 'node_' + str(node))]
                out = torch.squeeze(self.depth.forward_controller(depth_data, dropped_layers[:, 1]))
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.depth_adapter(out))
        # Dropped layers is b_size x 2 x 12
        # num_present_layers = torch.sum(dropped_layers, dim=-1)
        # layer_embeds = self.num_layer_embeds(num_present_layers)
        # Append the layer embeddings to the start, informing model about number of layers producing
        # each modality embedding
        agg_features = torch.stack(outlist, dim=1)
        #agg_features = torch.cat((layer_embeds, agg_features), dim=1)
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)

        out = self.encoder(agg_features) #bs x total_patches x 256
        out = torch.mean(out, dim=1)
        
        result_dict["early_fusion"] = self.output_head(out)['pred_mean'] # output_head returns a result of which we take the 'dist' key
        return result_dict, predicted_noise 
    



