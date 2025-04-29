import numpy as np
import torch
import torch.nn as nn
# in-folder debugging
# import adapters as adapters
# import backbones as backbones
# import output_head as output_head
# from vit_dev import ViT, LIMUBertEnc, TransformerDec

import models.adapters as adapters
import models.backbones as backbones
import models.output_head as output_head
from models.vit_dev import ViT, LIMUBertEnc, TransformerDec, TransformerEnc, positionalencoding1d
from torchvision import datasets, transforms, models
from torchsummary import summary
from einops import rearrange, repeat
from collections import deque
from models.PoseExpansion import PoseExpand
from models.layer_controller import LayerController, ConvLayerController
from scipy.spatial.transform import Rotation as R
from einops import rearrange, repeat
from torchvision.models import resnet18
from models.timm_vit import VisionTransformer
from timm.models.vision_transformer import Block, DropoutBlock

class MMFI_Early(nn.Module):
    def __init__(self, drop_layers_img = None, drop_layers_depth=None, layerdrop=0.0, vision_vit_layers=12, depth_vit_layers=12, valid_mods=['image', 'depth']):
        super(MMFI_Early, self).__init__()
        # Parameters used for multimodal fusion transformer
        dim_dec = 64 # USed to be 256
        depth_dec = 6 # previously 6
        heads = 4

        self.valid_mods = valid_mods

        if 'image' in valid_mods:
            # Initialize the vision backbone
            self.vision = VisionTransformer(
                patch_size=16, embed_dim=768, depth=vision_vit_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=nn.LayerNorm, layerdrop=layerdrop, drop_layers = drop_layers_img)
            if vision_vit_layers == 12:
                print(self.vision.load_state_dict(torch.load('MAE_Dropout_FT_Dropout.pth')['model'], strict=False))
                # Freeze the parameters, leave only the last layer unfrozen
                for param in self.vision.parameters():
                    param.requires_grad = False
                for block_num in range(11, 12):
                    for param in self.vision.blocks[block_num].parameters():
                        param.requires_grad = True
            self.vision_adapter = nn.Sequential(
                nn.Linear(768, 400),
                nn.ReLU(),
                nn.Linear(400, dim_dec)
            )
        if 'depth' in valid_mods:
            # Initialize the depth transformer, keeping the last two layers unfrozen
            self.depth = VisionTransformer(
                patch_size=16, embed_dim=768, depth=depth_vit_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=nn.LayerNorm, layerdrop=layerdrop, drop_layers = drop_layers_depth)
            #self.num_layer_embeds = nn.Embedding(13, dim_dec)
            if depth_vit_layers == 12:
                print(self.depth.load_state_dict(torch.load('MAE_Dropout_FT_Dropout.pth')['model'], strict=False))
                for param in self.depth.parameters():
                    param.requires_grad = False
                for block_num in range(11, 12):
                    for param in self.depth.blocks[block_num].parameters():
                        param.requires_grad = True
            self.depth_adapter = nn.Sequential(
                nn.Linear(768, 400),
                nn.ReLU(),
                nn.Linear(400, dim_dec)
            )
            
        self.encoder_cls = nn.Parameter(torch.rand(1, 1, dim_dec))
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        
        
 
        self.output_head = nn.Linear(dim_dec, 27)


        
    # Returns an dictionary of object results with (modality, node) as the keys
    # At test time, we override the 0.2 layerdrop by passing in drop_layers_img and drop_layers_depth
    # in the constructor
    def forward(self, data):
        
        
        outlist = []
        if 'image' in self.valid_mods:
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
            
            rgb_data = data['rgb'].cuda()
            batch_size, num_frames, channels, height, width = rgb_data.shape
            rgb_data = torch.reshape(rgb_data, (batch_size * num_frames, channels, height, width))
            out = self.vision.forward_train(rgb_data, dropped_layers_img) # Inflate batch size by the number of frames
            out = torch.squeeze(out)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            # out should now be (b_size x num_frames) x embed_dim
            out = self.vision_adapter(out) # run through the adapter, use this to gate how large I want to make the time encoder
            out = torch.reshape(out, (batch_size, num_frames, -1))
            outlist.append(out)
        if 'depth' in self.valid_mods:
            if self.training:
                dropped_layers_depth = (torch.rand(12) > self.depth.layerdrop_rate).int().cuda()
                if torch.rand(1).item() < 0.1:
                    dropped_layers_depth = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).cuda() # simulate entire modality dropout
            else:
                dropped_layers_depth = torch.ones(12).int().cuda()
            depth_data = data['depth'].cuda()
            batch_size, num_frames, channels, height, width = depth_data.shape
            depth_data = torch.reshape(depth_data, (batch_size * num_frames, channels, height, width))
            out = self.depth.forward_train(depth_data, dropped_layers_depth) # Inflate batch size by the number of frames
            out = torch.squeeze(out)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            # out should now be (b_size x num_frames) x embed_dim
            out = self.depth_adapter(out) # run through the adapter, use this to gate how large I want to make the time encoder
            out = torch.reshape(out, (batch_size, num_frames, -1))
            outlist.append(out)
       
        agg_features = torch.cat(outlist, dim=1)
        #agg_features = torch.cat(( self.encoder_cls.repeat(agg_features.shape[0], 1, 1), agg_features), dim=1)
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)

        out = self.encoder(agg_features)
        #out = out[:, 0]
        out = torch.mean(out, dim=1) # Perform mean pooling
        
        return self.output_head(out) # output_head returns a result of which we take the 'dist' key
        
    


# Passes downsampled input to a convolutional controller that outputs the number of layers per backbone
class Conv_MMFI_Controller(nn.Module):
    # Total layers dictates how many cumulative layers we want to limit our model to
    def __init__(self, total_layers=8):
        super(Conv_MMFI_Controller, self).__init__()

        dim_dec = 64
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
        
        self.vision_adapter = nn.Sequential(
            nn.Linear(768, 400),
            nn.ReLU(),
            nn.Linear(400, dim_dec)
        )
        self.depth_adapter = nn.Sequential(
            nn.Linear(768, 400),
            nn.ReLU(),
            nn.Linear(400, dim_dec)
        )
 
        self.output_head = nn.Linear(dim_dec, 27)


       
    # Returns an dictionary of object results with (modality, node) as the keys
    def forward(self, data, controller_temperature=1):




        outlist = []
        # dropped layers is now b_size x 30 x 2 x 12
        dropped_layers, predicted_noise = self.controller(data, controller_temperature)
        b, f, n, l = dropped_layers.shape
        dropped_layers = torch.reshape(dropped_layers, (-1, n, l))
        if 'rgb' in data.keys():
            rgb_data = data['rgb'].cuda()
            batch_size, num_frames, channels, height, width = rgb_data.shape
            rgb_data = torch.reshape(rgb_data, (batch_size * num_frames, channels, height, width))
            out = self.vision.forward_controller(rgb_data, dropped_layers[:, 0]) # Inflate batch size by the number of frames
            out = torch.squeeze(out)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            # out should now be (b_size x num_frames) x embed_dim
            out = self.vision_adapter(out) # run through the adapter, use this to gate how large I want to make the time encoder
            out = torch.reshape(out, (batch_size, num_frames, -1))
            outlist.append(out)
        if 'depth' in data.keys():
            depth_data = data['depth'].cuda()
            batch_size, num_frames, channels, height, width = depth_data.shape
            depth_data = torch.reshape(depth_data, (batch_size * num_frames, channels, height, width))
            out = self.depth.forward_controller(depth_data, dropped_layers[:, 1]) # Inflate batch size by the number of frames
            out = torch.squeeze(out)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            # out should now be (b_size x num_frames) x embed_dim
            out = self.depth_adapter(out) # run through the adapter, use this to gate how large I want to make the time encoder
            out = torch.reshape(out, (batch_size, num_frames, -1))
            outlist.append(out)
       
        agg_features = torch.cat(outlist, dim=1)
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)

        out = self.encoder(agg_features)
        out = torch.mean(out, dim=1) # Perform mean pooling
        
        return self.output_head(out), predicted_noise # output_head returns a result of which we take the 'dist' key
        


        
        
# Not necessary since I am now predicting a class with logits and not a multivariate distribution

# # Passes downsampled input to a convolutional controller that outputs the number of layers per backbone
# class Conv_MMFI_Controller_Test_FLOPS(nn.Module):
#     # Total layers dictates how many cumulative layers we want to limit our model to
#     def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes, total_layers=8):
#         super(Conv_MMFI_Controller_Test_FLOPS, self).__init__()

#         dim_dec = 256
#         depth_dec = 6
#         heads = 4

#         # In this version, we do not use the dynamic layerdrop or set the drop layers, this is all done by the model itself

#         self.vision = VisionTransformer(
#             patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#             norm_layer=nn.LayerNorm, block_fn = Block)
 
#         self.depth = VisionTransformer(
#             patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#             norm_layer=nn.LayerNorm, block_fn = Block)

#         self.controller = ConvLayerController(total_layers=total_layers)
#         #self.num_layers_embeds = nn.Embedding(12, dim_dec)
#         self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        
#         self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
#         self.depth_adapter = adapters.Adapter(768, adapter_hidden_dim)
#         self.output_head = output_head.OutputHead()
#         self.valid_mods = valid_mods
#         self.valid_nodes = valid_nodes


       
#     # Returns an dictionary of object results with (modality, node) as the keys
#     def forward(self, data, controller_temperature=5):
        
#         result_dict = {}
#         outlist = []
#         dropped_layers, predicted_noise = self.controller(data, self.valid_mods, self.valid_nodes, controller_temperature)
#         # 64 x n_mods x 12
#         if 'zed_camera_left' in self.valid_mods:
#             for node in self.valid_nodes: 
#                 vision_data = data[('zed_camera_left', 'node_' + str(node))]
#                 out = torch.squeeze(self.vision.forward_controller(vision_data, dropped_layers[:, 0]))
#                 if (len(out.shape) == 1):
#                     out = torch.unsqueeze(out, dim=0)
#                 outlist.append(self.vision_adapter(out))
#         if 'realsense_camera_depth' in self.valid_mods:
#             for node in self.valid_nodes:
#                 depth_data = data[('realsense_camera_depth', 'node_' + str(node))]
#                 out = torch.squeeze(self.depth.forward_controller(depth_data, dropped_layers[:, 1]))
#                 if (len(out.shape) == 1):
#                     out = torch.unsqueeze(out, dim=0)
#                 outlist.append(self.depth_adapter(out))
#         # Dropped layers is b_size x 2 x 12
#         # num_present_layers = torch.sum(dropped_layers, dim=-1)
#         # layer_embeds = self.num_layer_embeds(num_present_layers)
#         # Append the layer embeddings to the start, informing model about number of layers producing
#         # each modality embedding
#         agg_features = torch.stack(outlist, dim=1)
#         #agg_features = torch.cat((layer_embeds, agg_features), dim=1)
#         b, n, d = agg_features.shape
#         agg_features += positionalencoding1d(d, n)

#         out = self.encoder(agg_features) #bs x total_patches x 256
#         out = torch.mean(out, dim=1)
        
#         result_dict["early_fusion"] = self.output_head(out)['pred_mean'] # output_head returns a result of which we take the 'dist' key
#         return result_dict, predicted_noise 
    


if __name__== "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_nodes = [1,2,3]
    valid_mods = ["zed_camera_left", 'range_doppler', 'mic_waveform', 'realsense_camera_depth']
    model = MMFI_Early(adapter_hidden_dim=256, valid_mods=valid_mods, valid_nodes=valid_nodes).to(device)
    data = {('zed_camera_left','node_1'): torch.rand(64,3,270,480).to(device),
            ('zed_camera_left','node_2'): torch.rand(64,3,270,480).to(device),
            ('zed_camera_left','node_3'): torch.rand(64,3,270,480).to(device),
            ('realsense_camera_depth', 'node_1'): torch.rand(64,120,160).to(device),
            ('realsense_camera_depth', 'node_2'): torch.rand(64,120,160).to(device),
            ('realsense_camera_depth', 'node_3'): torch.rand(64,120,160).to(device),
            ('range_doppler', 'node_1'):torch.rand(64,256,16).to(device),
            ('range_doppler', 'node_2'):torch.rand(64,256,16).to(device),
            ('range_doppler', 'node_3'):torch.rand(64,256,16).to(device),
            ('mic_waveform', 'node_1'):torch.rand(64,4,1056).to(device),
            ('mic_waveform', 'node_2'):torch.rand(64,4,1056).to(device),
            ('mic_waveform', 'node_3'):torch.rand(64,4,1056).to(device)}
    res = model(data)
    import ipdb; ipdb.set_trace()
    


