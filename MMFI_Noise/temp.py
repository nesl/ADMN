import torch
import torch.nn as nn
import time
from torchvision import transforms
from models.vit_dev import TransformerEnc, positionalencoding1d

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


# This might be computationally intensive, how much better than TE is it? Make this an ablation
class ConvLayerControllerOld(nn.Module):
    # Take the CLS or global pool tokens of all the modalities
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4, num_modalities = 2, total_layers=6):
        super(ConvLayerControllerOld, self).__init__()
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
            nn.Linear(embed_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Sigmoid() # 12 layers in each ViT, we want to generate a one-hot at the end
        )
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
    def forward(self, batched_data, valid_mods, valid_nodes, temp=5):
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

        if self.training:
            gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.1)
        else:
            gumbel_samples = logits
        
        # Discretizing by taking the top_k values, getting 1 for layers we want to keep
        discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0) # discretize, taking only the top-k and replacing the rest with 0 or 0.01
        discretized = torch.reshape(discretized, (B, -1, 12))
        discretized[:, :, 0] = 1

        gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
        logits = torch.reshape(logits, (B, -1, 12))

       
        # print('Image:', logits[0][0])
        # print('Depth:', logits[0][1])
        # print('Image:', discretized[0][0])
        # print('Depth:',  discretized[0][1])

        predicted_noise = self.noise_output(x) # b_size x 2 (img and depth)

        return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise * 5

# This might be computationally intensive, how much better than TE is it? Make this an ablation
class ConvLayerController(nn.Module):
    # Take the CLS or global pool tokens of all the modalities
    def __init__(self, embed_dim=256, depth=2, num_heads=4, mlp_ratio=1, num_modalities = 2, total_layers=6):
        super(ConvLayerController, self).__init__()
        # Downsample input to 100 x 100 and then pass through conv layers
        self.encoder_dict = nn.ModuleDict({
            'zed_camera_left': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10), stride=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(576, embed_dim)
            ),
            'realsense_camera_depth': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10), stride=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(576, embed_dim)
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
            nn.Linear(embed_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Sigmoid() # 12 layers in each ViT, we want to generate a one-hot at the end
        )
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
    def forward(self, batched_data, valid_mods, valid_nodes, temp=5):
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

        if self.training:
            gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.1)
        else:
            gumbel_samples = logits
        
        # Discretizing by taking the top_k values, getting 1 for layers we want to keep
        discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0) # discretize, taking only the top-k and replacing the rest with 0 or 0.01
        discretized = torch.reshape(discretized, (B, -1, 12))
        discretized[:, :, 0] = 1

        gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
        logits = torch.reshape(logits, (B, -1, 12))
        # # if not self.training: # Only use this during testing
        # #     self.logits_memory.append(logits.detach().cpu())
       
        # print('Image:', logits[0][0])
        # print('Depth:', logits[0][1])
        # print('Image:', discretized[0][0])
        # print('Depth:',  discretized[0][1])

        predicted_noise = self.noise_output(x) # b_size x 2 (img and depth)
        # # if gumbel_samples.requires_grad:
        # #     gumbel_samples.register_hook(self.print_grad)
        # # TODO Remove this
        # #temporary_discretized = torch.stack([torch.ones(B, 12), torch.ones(B, 12)], dim=1).cuda()
        return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise * 5
        # #return gumbel_samples + (temporary_discretized - gumbel_samples).detach(), predicted_noise * 5





network = ConvLayerControllerOld().cuda()

dummy = {
    ('realsense_camera_depth', 'node_1'): torch.zeros(1, 3, 256, 256).cuda(),
    ('realsense_camera_depth', 'node_2'): torch.zeros(1, 3, 256, 256).cuda(),
    ('realsense_camera_depth', 'node_3'): torch.zeros(1, 3, 256, 256).cuda(),
    ('zed_camera_left', 'node_1'): torch.zeros(1, 3, 256, 256).cuda(),
    ('zed_camera_left', 'node_2'): torch.zeros(1, 3, 256, 256).cuda(),
    ('zed_camera_left', 'node_3'): torch.zeros(1, 3, 256, 256).cuda(),
}
network.eval()
start = time.time()
for i in range(2701):
    # img and depth, 3 nodes

    network(dummy, ['zed_camera_left', 'realsense_camera_depth'], [1, 2, 3])
print(time.time() - start)