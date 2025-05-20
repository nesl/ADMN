import torch
import torch.nn as nn
from models.vit_dev import TransformerEnc, positionalencoding1d
from torchvision import transforms
import numpy as np
from einops import rearrange

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

# x is b_size x 24, assume k is the number of audio layers, vision is 3x audio
# If not valid
def get_top_k_unequal(x, k=8, zero_value=0):
    budget_max = k
    k = min(k, 24)
    top_k_indices = torch.topk(x, k, dim=1).indices # Returns top_k (b_size x k) 
    for batch_idx in range(top_k_indices.shape[0]):
        budget_count = 0
        for i in range(k):
            if budget_count <= budget_max - 3:
                budget_count += 3 if top_k_indices[batch_idx][i] < 12 else 1 # vision contributes 3
            elif budget_count < budget_max and top_k_indices[batch_idx][i] >= 12:
                budget_count += 1 # Greater than k-3 but less than k, fill with ones
            else:
                top_k_indices[batch_idx][i] = 0 # Set to index zero, index 0 is always chosen anyways
    result = torch.full(x.shape, zero_value).cuda()
    return result.scatter_(1, top_k_indices, 1) # hopefully scatter works w repeats




def sample_gumbel(shape, scale):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + 1e-12) + 1e-12) * scale

# Perform gumbel softmax sampling with specified temperature and scale
def gumbel_softmax_sample(logits, temperature, scale=1):
    y = logits + sample_gumbel(logits.size(), scale)
    return nn.functional.softmax(y / temperature, dim=-1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-0.005, 0.005)

    
# This is the Controller that allocates layers among modalities in accordance to modality quality
class ConvLayerController(nn.Module):
    
    # Total layer refers to the total available compute budget
    # TODO CHANGE THIS back to 256
    def __init__(self, embed_dim=512, depth=4, num_heads=4, mlp_ratio=4, num_modalities = 2, total_layers=6):
        super(ConvLayerController, self).__init__()

        self.encoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'audio': nn.Sequential(
                transforms.Resize((128, 512)),
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 6)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(7488, embed_dim)
            )
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
            nn.Linear(250, 4),
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
    def forward(self, batched_data, valid_mods, temp=1, discretization_method = 'admn'):
        audio_data, img_data, labels = batched_data
        conv_embeds = []
        if 'image' in valid_mods:
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w')
            out = self.encoder_dict['image'](img_data)
            out = rearrange(out, '(b s) e-> b s e', b = audio_data.shape[0])
            conv_embeds.append(out)
        if 'audio' in valid_mods:
            out = self.encoder_dict['audio'](audio_data)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            out = torch.unsqueeze(out, dim=1)
            conv_embeds.append(out)

        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output

        logits = self.output_head(x) # logits are of shape B_size x 24 \
        # First logit for each modality will ALWAYS be chosen, set it to -99 to avoid influencing the softmax too heavily
        
        logits[:, 0] = -99
        logits[:, 12] = -99

        # Get the predicted noise for each of the modalities
        predicted_noise = self.noise_output(x) # b_size x 2 (img and depth)

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
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
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
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
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
            print('Audio:',  discretized[0][1])
            return discretized, predicted_noise
        else:
            raise Exception('Invalid discretization')



# This is the Controller that allocates layers among modalities in accordance to modality quality
class ConvLayerControllerUnequal(nn.Module):
    
    # Total layer refers to the total available compute budget
    # TODO CHANGE THIS back to 256
    def __init__(self, embed_dim=512, depth=4, num_heads=4, mlp_ratio=4, num_modalities = 2, total_layers=6):
        super(ConvLayerControllerUnequal, self).__init__()

        self.encoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'audio': nn.Sequential(
                transforms.Resize((128, 512)),
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 6)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(7488, embed_dim)
            )
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        self.additional_layers = total_layers - 4 # how many layers we are allocating, vision is 3 layers, audio is 1
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 200, bias=False),
            nn.ReLU(),
            nn.Linear(200, 12 * num_modalities, bias=False) # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.noise_output = nn.Sequential(
            nn.Linear(embed_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 4),
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
    def forward(self, batched_data, valid_mods, temp=1, discretization_method = 'admn'):
        audio_data, img_data, labels = batched_data
        conv_embeds = []
        if 'image' in valid_mods:
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w')
            out = self.encoder_dict['image'](img_data)
            out = rearrange(out, '(b s) e-> b s e', b = audio_data.shape[0])
            conv_embeds.append(out)
        if 'audio' in valid_mods:
            out = self.encoder_dict['audio'](audio_data)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            out = torch.unsqueeze(out, dim=1)
            conv_embeds.append(out)

        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output

        logits = self.output_head(x) # logits are of shape B_size x 24 \
        # First logit for each modality will ALWAYS be chosen, set it to -99 to avoid influencing the softmax too heavily
        
        logits[:, 0] = -99
        logits[:, 12] = -99

        # Get the predicted noise for each of the modalities
        predicted_noise = self.noise_output(x) # b_size x 2 (img and depth)

        if discretization_method == 'admn':
            if self.training:
                gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.1)
            else: # If this is during inference, we don't do any gumbel softmax sampling
                gumbel_samples = logits
            discretized = get_top_k_unequal(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1 # Set the first layer to always chosen
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            print('Image:', logits[0][0])
            print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'straight_through': # No softmax sampling used
            gumbel_samples = logits
            discretized = get_top_k_unequal(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'progressive': # In theory this would require us to progressively adjust the temperature w gumbel softmax only
            if self.training:
                sampler = SubsetOperator(self.additional_layers, tau=temp, hard=False)
                discretized = sampler(logits)
            else:
                discretized = get_top_k_unequal(logits, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return discretized, predicted_noise
        else:
            raise Exception('Invalid discretization')


class Conv_Controller_AE(nn.Module):
    # Total layer refers to the total available compute budget
    def __init__(self, embed_dim=512, depth=4, num_heads=4, mlp_ratio=4):
        super(Conv_Controller_AE, self).__init__()
        # Do the respective resizes before the AE function
        self.encoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'audio': nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 6)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(7488, embed_dim)
            )
        })
        self.decoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(14, 14)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(12, 8), stride=(3, 2)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5)),
            ),
            'audio': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(10, 8), stride=(2, 3)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(7, 8), stride=(3, 5)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5)),
            )
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        
    def forward(self, batched_data, valid_mods):
        audio_data, img_data, labels = batched_data
        conv_embeds = []
        if 'image' in valid_mods:
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w')      
            out = self.encoder_dict['image'](img_data)
            out = rearrange(out, '(b s) e-> b s e', b = audio_data.shape[0])
            conv_embeds.append(out)
        if 'audio' in valid_mods:
            out = self.encoder_dict['audio'](audio_data)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            out = torch.unsqueeze(out, dim=1)
            conv_embeds.append(out)

        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output
        x = torch.reshape(x, (-1, 1, 16, 32))
        
        img_recon = self.decoder_dict['image'](x)
        audio_recon = self.decoder_dict['audio'](x)
        return img_recon, audio_recon




# This is the Controller that allocates layers among modalities in accordance to modality quality
class AdaMML_Modality_Selector(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4):
        super(AdaMML_Modality_Selector, self).__init__()
        self.encoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'audio': nn.Sequential(
                transforms.Resize((128, 512)),
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 6)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(7488, embed_dim)
            )
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

    # Temperature define peakiness of the gumbel softmax
    def forward(self, batched_data):
        audio_data, img_data, labels = batched_data
        conv_embeds = []
        img_data = rearrange(img_data, 'b s c h w -> (b s) c h w')
        out = self.encoder_dict['image'](img_data)
        out = rearrange(out, '(b s) e-> b s e', b = audio_data.shape[0])
        conv_embeds.append(out)
        out = self.encoder_dict['audio'](audio_data)
        if (len(out.shape) == 1):
            out = torch.unsqueeze(out, dim=0)
        out = torch.unsqueeze(out, dim=1)
        conv_embeds.append(out)

        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output

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