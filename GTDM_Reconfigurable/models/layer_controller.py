import torch
import torch.nn as nn
from models.vit_dev import TransformerEnc, positionalencoding1d
from torchvision import transforms
import numpy as np

EPSILON = np.finfo(np.float32).tiny

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

def softmax_k(x, temp=0.005, k=8):
    
    weight = torch.zeros_like(x)
    for i in range(k):
        weight_rounded = weight + (torch.round(weight) - weight).detach()
        logw = torch.log10((1 - weight_rounded) + 1e-16) # Fixes precision issues, can start out with smaller temperature
        scaled = (x + logw) / temp
        max_arr = torch.max(scaled, dim=-1, keepdim=True)[0]
        normalized = scaled - max_arr
        weight += torch.exp(normalized) / torch.sum(torch.exp(normalized), dim=-1, keepdim=True)
    if torch.sum(weight != weight) != 0:
        import pdb; pdb.set_trace()
    return weight

def get_top_k(x, k=8, zero_value=0):
    top_k_indices = torch.topk(x, k, dim=1).indices
    result = torch.full(x.shape, zero_value).cuda()
    return result.scatter_(1, top_k_indices, 1)

def gumbel_sigmoid(logits, tau=1):
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


def sample_gumbel(shape, scale):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + 1e-12) + 1e-12) * scale

def gumbel_softmax_sample(logits, temperature, scale=1):
    y = logits + sample_gumbel(logits.size(), scale)
    return nn.functional.softmax(y / temperature, dim=-1)

class LayerController(nn.Module):
    # Take the CLS or global pool tokens of all the modalities
    # Total layers will include the first two layers that are default included in the ViT for the controller
    def __init__(self, embed_dim=768, depth=4, num_heads=4, mlp_ratio=4, num_modalities = 2, total_layers=6):
        super(LayerController, self).__init__()
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        self.additional_layers = total_layers - 2 
        self.project_head = nn.Linear(768, embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 12 * num_modalities) # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.logits_memory = [] # Used during inference only
        self.gradient_sum = torch.zeros((2, 12))

    def decrease_model_layers(self, min_layers):
        min_additional_layers = min_layers - 2
        if self.additional_layers > min_additional_layers:
            self.additional_layers -= 1
    # Used during backward hook to see the gradient values, will help if I want to implement gradient clipping to prevent weird behavior
    def print_grad(self, grad):
        #grad = torch.clip(grad, -0.2, 0)
        print(grad[0])
        return grad
        #self.gradient_sum += torch.sum(grad, dim=0).detach().cpu()
    def forward(self, x):
        B = x.shape[0]
        #x = self.project_head(x)
        cls_tokens = self.cls.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.combiner_encoder(x)[:, 0]
  
        logits = self.output_head(x) # logits are of shape B_size x 24 
        probs = torch.nn.functional.softmax(logits, dim=-1) # convert to probabilities using a single softmax
        zero_value = 0.01 if self.training else 0 # 0.1 is fine if im not letting downstream layers adapt to this
        discretized = get_top_k(probs, k=self.additional_layers, zero_value=zero_value) # discretize, taking only the top-k and replacing the rest with 0 or 0.01
        # Use 0.01 for training since that allows us to get some gradients, use 0 for inference and validation
        # TODO experiment whether 0.01 is ideal or if it is too small 

        probs = torch.reshape(probs, (B, -1, 12))
        if probs.requires_grad:
            probs.register_hook(self.print_grad)
        discretized = torch.reshape(discretized, (B, -1, 12))
        logits = torch.reshape(logits, (B, -1, 12))
        # if not self.training: # Only use this during testing
        #     self.logits_memory.append(logits.detach().cpu())
        if self.training:
            print('Image:', logits[0][0])
            print('Depth:', logits[0][1])
        else:
            print('Image:', discretized[0][0])
            print('Depth:', discretized[0][1])

        return probs + (discretized - probs).detach()
        #return self.logits

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-0.005, 0.005)
        #m.bias.data.fill_(0.0)
    
# This might be computationally intensive, how much better than TE is it? Make this an ablation
class ConvLayerController(nn.Module):
    # Take the CLS or global pool tokens of all the modalities
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
        self.additional_layers = total_layers - 2 # how many layers we are allocating
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 200, bias=False),
            nn.ReLU(),
            nn.Linear(200, 12 * num_modalities, bias=False) # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.noise_output = nn.Sequential(
            nn.Linear(embed_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 2),
            nn.Sigmoid() # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.output_head.apply(init_weights)
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
    def forward(self, batched_data, valid_mods, valid_nodes, temp=5, discretization_method = 'admn'):
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
        
        cls_tokens = self.cls.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output
        logits = self.output_head(x) # logits are of shape B_size x 24 
        logits[:, 0] = -99
        logits[:, 12] = -99

        predicted_noise = self.noise_output(x) # b_size x 2 (img and depth)
        if discretization_method == 'admn':
            if self.training:
                gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.1)
            else:
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
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise * 5
        elif discretization_method == 'straight_through':
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
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise * 5
        elif discretization_method == 'progressive':
            if self.training:
                sampler = SubsetOperator(self.additional_layers, tau=temp, hard=False)
                discretized = sampler(logits)
            else:
                discretized = get_top_k(logits, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            #gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Depth:',  discretized[0][1])
            return discretized, predicted_noise * 5
        else:
            raise Exception('Invalid discretization')
        
        # # Discretizing by taking the top_k values, getting 1 for layers we want to keep
        #  # discretize, taking only the top-k and replacing the rest with 0 or 0.01
        
       
        #





if __name__ == '__main__':
    test = torch.rand(64, 24)
    res = softmax_k(test)
    import pdb; pdb.set_trace()
