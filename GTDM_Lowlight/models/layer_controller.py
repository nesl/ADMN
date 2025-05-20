import torch
import torch.nn as nn
from models.vit_dev import TransformerEnc, positionalencoding1d
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

EPSILON = np.finfo(np.float32).tiny

# This does gumbel softmax sampling without using straight through estimator
class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

# Used to sample top k, returning 1's in the top k indices and 0's otherwise
def get_top_k(x, k=8, zero_value=0):
    top_k_indices = torch.topk(x, k, dim=1).indices
    result = torch.full(x.shape, zero_value).cuda()
    return result.scatter_(1, top_k_indices, 1)


def sample_gumbel(shape, scale):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + 1e-12) + 1e-12) * scale

# Perform gumbel softmax sampling with specified temperature and scale
def gumbel_softmax_sample(logits, temperature, scale=1):
    y = logits + sample_gumbel(logits.size(), scale)
    return nn.functional.softmax(y / temperature, dim=-1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-0.005, 0.005) # Kinda works but is unstable w.r.t seed, can we do better
        #m.weight.data.uniform_(-0.001, 0.001) # is this better?

    
# This is the Controller that allocates layers among modalities in accordance to modality quality
class ConvLayerController(nn.Module):
    
    # Total layer refers to the total available compute budget
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4, num_modalities = 2, total_layers=6):
        super(ConvLayerController, self).__init__()

        # Downsample input to 100 x 100 and then pass through conv layers
        self.encoder_dict = nn.ModuleDict({
            'zed_camera_left': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'realsense_camera_depth': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        self.additional_layers = total_layers - 2 # how many layers we are allocating, first layer is always 1
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 200, bias=False),
            nn.ReLU(),
            nn.Linear(200, 12 * num_modalities, bias=False) # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.noise_output = nn.Sequential(
            nn.Linear(embed_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 4)
        )
        self.output_head.apply(init_weights) # init output head weights to be very small, this forces it to be responsive to the noise embedding value
        self.logits_memory = [] # Used during inference only
        self.grad_accum = None

    # Used during backward hook to see the gradient values, will help if I want to implement gradient clipping to prevent weird behavior
    # def print_grad(self, grad):
    #     #print("THE GRADIENT IS", grad[0])
    #     if self.grad_accum is None:
    #         self.grad_accum = torch.mean(grad, dim=0)
    #     else:
    #         self.grad_accum += torch.mean(grad, dim=0)
    #     #return torch.clip(grad, -0.1, 0.1)
    #     return grad 

    # Temperature define peakiness of the gumbel softmax
    def forward(self, batched_data, valid_mods, valid_nodes, temp=1, discretization_method = 'admn'):
        # Get all the convolutional embeds of each modality of each node (6)
        conv_embeds = []
        if 'zed_camera_left' in valid_mods:
            for node in valid_nodes:
                key = ('zed_camera_left', 'node_' + str(node))
                out = self.encoder_dict[key[0]](batched_data[key])
                conv_embeds.append(out)
        if 'realsense_camera_depth' in valid_mods:
            for node in valid_nodes:
                key = ('realsense_camera_depth', 'node_' + str(node))
                out = self.encoder_dict[key[0]](batched_data[key])
                conv_embeds.append(out)
        conv_embeds = torch.stack(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output

        logits = self.output_head(x) # logits are of shape B_size x 24 \
        # First logit for each modality will ALWAYS be chosen, set it to -99 to avoid influencing the softmax too heavily
        logits[:, 0] = -99
        logits[:, 12] = -99

        # Get the predicted gamma for each of the modalities
        pred_gamma = self.noise_output(x) # b_size x 2 (img and depth)

        if discretization_method == 'admn':
            if self.training:
                gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.1)
            else: # If this is during inference, we don't do any gumbel softmax sampling
                gumbel_samples = logits
            discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1 # Set the first layer to always chosen
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            print('Image:', logits[0][0])
            print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Depth:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), pred_gamma
        elif discretization_method == 'straight_through': # No softmax sampling used
            gumbel_samples = logits
            discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Depth:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), pred_gamma
        elif discretization_method == 'progressive': # In theory this would require us to progressively adjust the temperature w gumbel softmax only
            if self.training:
                sampler = SubsetOperator(self.additional_layers, tau=temp, hard=False)
                discretized = sampler(logits)
            else:
                discretized = get_top_k(logits, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Depth:',  discretized[0][1])
            return discretized, pred_gamma
        else:
            raise Exception('Invalid discretization')


class Conv_Controller_AE(nn.Module):
    # Total layer refers to the total available compute budget
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4):
        super(Conv_Controller_AE, self).__init__()
        # Do the respective resizes before the AE function
        self.encoder_dict = nn.ModuleDict({
            'zed_camera_left': nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'realsense_camera_depth': nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
        })
        self.decoder_dict = nn.ModuleDict({
            'zed_camera_left': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(14, 14)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(12, 12), stride=(3, 3)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5)),
            ),
            'realsense_camera_depth': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(14, 14)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(12, 12), stride=(3, 3)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5)),
            ),
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        
    def forward(self, batched_data, valid_mods, valid_nodes):
        conv_embeds = []
        if 'zed_camera_left' in valid_mods:
            for node in valid_nodes:
                key = ('zed_camera_left', 'node_' + str(node))
              
                out = self.encoder_dict[key[0]](batched_data[key])
                conv_embeds.append(out)

        if 'realsense_camera_depth' in valid_mods:
            for node in valid_nodes:
                key = ('realsense_camera_depth', 'node_' + str(node))
                out = self.encoder_dict[key[0]](batched_data[key])
                conv_embeds.append(out)
        conv_embeds = torch.stack(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output
        x = torch.reshape(x, (-1, 1, 16, 16))
        
        img_recon = self.decoder_dict['zed_camera_left'](x)
        depth_recon = self.decoder_dict['realsense_camera_depth'](x)
        return img_recon, depth_recon
    


# the modality selector for AdaMML
class AdaMML_Modality_Selector(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4):
        super(AdaMML_Modality_Selector, self).__init__()
        
        # Downsample input to 100 x 100 and then pass through conv layers
        self.encoder_dict = nn.ModuleDict({
            'zed_camera_left': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'realsense_camera_depth': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
        })

        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))

        # output head for the 3-way modality selector
        self.mod_sel_head = nn.Sequential(
            nn.Linear(embed_dim, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 3, bias=False)  # 3 classes: Img only / Dep only / Both
        )

        self.temperature = 5.0

    def forward(self, batched_data, valid_nodes):
        conv_embeds = []

        for modality in ['zed_camera_left', 'realsense_camera_depth']:
            for node in valid_nodes:
                key = (str(modality), 'node_' + str(node))
                out = self.encoder_dict[key[0]](batched_data[key])
                conv_embeds.append(out)
        conv_embeds = torch.stack(conv_embeds, dim=1) # B * 6 * embed_dim
        B = conv_embeds.shape[0]

        cls_tokens = self.cls.expand(B, -1, -1)  # B*1 * embed_dim
        x = torch.cat((cls_tokens, conv_embeds), dim=1) # B * 7 * embed_dim
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output B * embed_dim

        logits = self.mod_sel_head(x) # B * 3

        tau = self.temperature

        if self.training:
            soft = F.gumbel_softmax(logits, tau=tau, hard=False) # B * 3
            # construct had one-hot
            idx = torch.argmax(soft, dim=-1)
            hard = F.one_hot(idx, num_classes=3).float()

            # STE
            samp = soft + (hard - soft).detach()
        else:
            # use hard argmax for inference
            idx = torch.argmax(logits, dim=-1) # B
            samp = F.one_hot(idx, num_classes=3).float().to(logits.device) # B * 3

        # project the 3-way one-hot to binary decision
        mask = torch.zeros(B, 2, device=logits.device)
        mask[:, 0] = samp[:, 0] + samp[:, 1]
        mask[:, 1] = samp[:, 2] + samp[:, 1]

        return samp, mask, logits

