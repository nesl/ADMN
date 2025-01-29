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

class GTDM_Early(nn.Module):
    def __init__(self, adapter_hidden_dim, 
                 valid_mods, 
                 valid_nodes, 
                 drop_layers_img = None, 
                 drop_layers_depth=None, 
                 drop_layers_mmwave=None, 
                 layerdrop=0.0, 
                 vision_vit_layers=12, 
                 depth_vit_layers=12, 
                 mmWave_vit_layers=12):
        super(GTDM_Early, self).__init__()
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



        # Initialize the depth transformer, keeping the last two layers unfrozen
        self.mmWave = VisionTransformer(img_size=(256, 16),
            patch_size=16, embed_dim=768, depth=mmWave_vit_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, layerdrop=layerdrop, drop_layers = drop_layers_mmwave)

        if mmWave_vit_layers == 12:
            state_dict = torch.load('./logs/mmWave_Model/last.pt')
            new_state_dict = {}
            for key in state_dict:
                if 'mmWave' in key:
                    new_state_dict[key[7:]] = state_dict[key]

            print(self.mmWave.load_state_dict(new_state_dict, strict=False))
            for param in self.mmWave.parameters():
                param.requires_grad = False
            # for param in self.mmWave.blocks[0].parameters():
            #     param.requires_grad = True
            # for param in self.mmWave.blocks[6].parameters():
            #     param.requires_grad = True
            # for param in self.mmWave.blocks[10].parameters():
            #     param.requires_grad = True
            for param in self.mmWave.blocks[11].parameters():
                param.requires_grad = True

        # Use encoder to combine the information
        #self.fusion_cls = nn.Parameter(torch.randn(1, dim_dec))
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.depth_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.mmWave_adapter = adapters.Adapter(768, adapter_hidden_dim)
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
                if torch.rand(1).item() < 0.2:
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
                if torch.rand(1).item() < 0.2:
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
       

        if 'range_doppler' in self.valid_mods:
            print("MMWAVE")
            # 1 indicates that we keep this layer, 0 indicates that it is dropped
            if self.training:
                # Initialize ONE noise vector for the entire batch and for all distributed cameras
                # This is bc we speed up training by avoiding computing zeroed out info
                dropped_layers_mmWave = (torch.rand(12) > self.mmWave.layerdrop_rate).int().cuda()
                if torch.rand(1).item() < 0.2:
                    dropped_layers_mmWave = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).cuda() # simulate entire modality dropout
            else:
                # If we are not training, do not drop any layers
                dropped_layers_mmWave = torch.ones(12).int().cuda()
            for node in self.valid_nodes:
                node = str(node)
                data_transformed = data[('range_doppler', 'node_' + str(node))]
                out = self.mmWave.forward_train(data_transformed, dropped_layers_mmWave)
                out = torch.squeeze(out)
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.mmWave_adapter(out))

        agg_features = torch.stack(outlist, dim=1)
        #agg_features = torch.cat((cls_tokens, agg_features), dim=1)
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)

        out = self.encoder(agg_features)
        out = torch.mean(out, dim=1) # Perform mean pooling
        
        result_dict["early_fusion"] = self.output_head(out) # output_head returns a result of which we take the 'dist' key
        return result_dict
    


# DEPCRECIATED SINCE USING CLS OF LAYER ONE VIT OUTPUT IS PROBABLY BAD
class GTDM_Controller(nn.Module):
    # Total layers dictates how many cumulative layers we want to limit our model to
    def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes, drop_layers = None, layerdrop=0.0, total_layers=6):
        super(GTDM_Controller, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_vit = 128
        dim_dec = 256
        depth_vit = 6
        depth_dec = 6
        heads = 8
        dropout = 0.2
        emb_dropout = 0.2



        # In this version, we do not use the dynamic layerdrop or set the drop layers, this is all done by the model itself
        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm)
 

        self.depth = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm)

        self.controller = LayerController(total_layers=total_layers)
        
        
        # Use encoder to combine the information, 3 layers to have better crossmodal learning
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_vit//heads, mlp_dim=3*dim_dec, dropout=emb_dropout)
        self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.depth_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.output_head = output_head.OutputHead()

        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes


       
    # Returns an dictionary of object results with (modality, node) as the keys
    def forward(self, data):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result_dict = {}
        outlist = []
        first_layer_out = []
        if 'zed_camera_left' in self.valid_mods:
            for node in self.valid_nodes:
                node = str(node)
                data_transformed = data[('zed_camera_left', 'node_' + str(node))]
                out = self.vision.run_N_layers(data_transformed, 1)
                first_layer_out.append(out)
        if 'realsense_camera_depth' in self.valid_mods:
            for node in self.valid_nodes:
                node = str(node)
                depth_data = data[('realsense_camera_depth', 'node_' + str(node))]
                out = self.depth.run_N_layers(depth_data, 1)
                first_layer_out.append(out)
 
        aggregated_cls_tokens = torch.stack([embed[:, 0] for embed in first_layer_out], dim=1)

        dropped_layers = self.controller(aggregated_cls_tokens)
        # 64 x n_mods x 12
        #print('Image layers: ', 12 - torch.sum(dropped_layers, dim=-1)[:, 0])
        if 'zed_camera_left' in self.valid_mods:
            for node in self.valid_nodes: 
                out = torch.squeeze(self.vision.run_remaining_layers(first_layer_out.pop(0), 1, dropped_layers[:, 0]))
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.vision_adapter(out))
        if 'realsense_camera_depth' in self.valid_mods:
            for node in self.valid_nodes:
                out = torch.squeeze(self.depth.run_remaining_layers(first_layer_out.pop(0), 1, dropped_layers[:, 1]))
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.depth_adapter(out))
      
        agg_features = torch.stack(outlist, dim=1)
        b, n, _ = agg_features.shape

        ##agg_features = agg_features + self.final_pos_embedding
        out = self.encoder(agg_features) #bs x total_patches x 256
        out = torch.mean(out, dim=1)
        
        result_dict["early_fusion"] = self.output_head(out) # output_head returns a result of which we take the 'dist' key
        return result_dict 
    


# Passes downsampled input to a convolutional controller that outputs the number of layers per backbone
class Conv_GTDM_Controller(nn.Module):
    # Total layers dictates how many cumulative layers we want to limit our model to
    def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes, total_layers=8):
        super(Conv_GTDM_Controller, self).__init__()

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

        self.mmWave = VisionTransformer(img_size=(256, 16),
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, block_fn = Block)
        
        self.controller = ConvLayerController(total_layers=total_layers, num_modalities=3)
        #self.num_layers_embeds = nn.Embedding(12, dim_dec)
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_dec//heads, mlp_dim=3*dim_dec)
        
        self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.depth_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.mmWave_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.output_head = output_head.OutputHead()
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes


       
    # Returns an dictionary of object results with (modality, node) as the keys
    def forward(self, data, controller_temperature=5):
        
        result_dict = {}
        outlist = []
        dropped_layers, predicted_noise = self.controller(data, self.valid_mods, self.valid_nodes, controller_temperature)
        #dropped_layers, predicted_noise = torch.ones(1, 2, 12).cuda(), torch.ones(1, 2).cuda()
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
        if 'range_doppler' in self.valid_mods:
            for node in self.valid_nodes:
                mmWave_data = data[('range_doppler', 'node_' + str(node))]
                out = torch.squeeze(self.mmWave.forward_controller(mmWave_data, dropped_layers[:, 2]))
                if (len(out.shape) == 1):
                    out = torch.unsqueeze(out, dim=0)
                outlist.append(self.mmWave_adapter(out))
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
        
        result_dict["early_fusion"] = self.output_head(out) # output_head returns a result of which we take the 'dist' key
        return result_dict, predicted_noise 
    





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
    



# Passes downsampled input to a convolutional controller that outputs the number of layers per backbone
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
    

# Depreciated
class GTDM_CLS(nn.Module):
    # Total layers dictates how many cumulative layers we want to limit our model to
    def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes, drop_layers = None, layerdrop=0.0, total_layers=6):
        super(GTDM_Controller, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_vit = 128
        dim_dec = 256
        depth_vit = 6
        depth_dec = 6
        heads = 8
        dropout = 0.2
        emb_dropout = 0.2



        # In this version, we do not use the dynamic layerdrop or set the drop layers, this is all done by the model itself

        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm)
 

        self.depth = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm)

        self.controller = LayerController(total_layers=total_layers)
        
       
        self.encoder = TransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_vit//heads, mlp_dim=3*dim_dec, dropout=emb_dropout)
        self.vision_adapter = adapters.Adapter(768, adapter_hidden_dim)
        self.depth_adapter = adapters.Adapter(768, adapter_hidden_dim)
 
     
        self.output_head = output_head.OutputHead()
        self.noise_predictor = nn.Sequential(
            nn.Linear(768, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes


       
    # Returns an dictionary of object results with (modality, node) as the keys
    def forward(self, data):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result_dict = {}
        outlist = []
        first_layer_out = []
        if 'zed_camera_left' in self.valid_mods:
            for node in self.valid_nodes:
                node = str(node)
                data_transformed = data[('zed_camera_left', 'node_' + str(node))]
                out = self.vision.run_N_layers(data_transformed, 1)
                first_layer_out.append(out)
        if 'realsense_camera_depth' in self.valid_mods:
            for node in self.valid_nodes:
                node = str(node)
                depth_data = data[('realsense_camera_depth', 'node_' + str(node))]
                out = self.depth.run_N_layers(depth_data, 1)
                first_layer_out.append(out)
 
        aggregated_cls_tokens = torch.stack([embed[:, 0] for embed in first_layer_out], dim=1)

        
        return result_dict 


if __name__== "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_nodes = [1,2,3]
    valid_mods = ["zed_camera_left", 'range_doppler', 'mic_waveform', 'realsense_camera_depth']
    model = GTDM_Early(adapter_hidden_dim=256, valid_mods=valid_mods, valid_nodes=valid_nodes).to(device)
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
    


