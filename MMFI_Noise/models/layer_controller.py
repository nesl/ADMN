import torch
import torch.nn as nn
from models.vit_dev import TransformerEnc, positionalencoding1d
from torchvision import transforms
import torch.nn.functional as F

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
    def __init__(self, embed_dim=256, depth=6, num_heads=4, mlp_ratio=4, num_modalities = 2, total_layers=6):
        super(ConvLayerController, self).__init__()
        # Downsample input to 100 x 100 and then pass through conv layers
        self.encoder_dict = nn.ModuleDict({
            'rgb': nn.Sequential(
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
            'depth': nn.Sequential(
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
            nn.Linear(embed_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 2),
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
    def forward(self, batched_data, temp=5):
        # Get all the convolutional embeds of each modality of each node (6)
        conv_embeds = []
        if 'rgb' in batched_data.keys():
            b, n, c, h, w = batched_data['rgb'].shape
            out = self.encoder_dict['rgb'](torch.reshape(batched_data['rgb'][:, 0:3], (-1, c, h, w))) # Take only the noise of the first three frames
            conv_embeds.append(torch.reshape(out, (b, 3, -1)))
        if 'depth' in batched_data.keys():
            b, n, c, h, w = batched_data['depth'].shape
            out = self.encoder_dict['depth'](torch.reshape(batched_data['depth'][:, 0:3], (-1, c, h, w))) # Take only the noise of the first three frames
            conv_embeds.append(torch.reshape(out, (b, 3, -1)))
        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        x = conv_embeds
        #cls_tokens = self.cls.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(x.shape[-1], x.shape[1])
        #x = self.combiner_encoder(x)[:, 0] # Get CLS output
        x = torch.mean(self.combiner_encoder(x), dim=1)
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

       
        print('Image:', logits[0][0])
        print('Depth:', logits[0][1])
        print('Image:', discretized[0][0])
        print('Depth:',  discretized[0][1])

        predicted_noise = self.noise_output(x) # b_size x 2 (img and depth) x 12

        gumbel_samples = torch.unsqueeze(gumbel_samples, dim=1).repeat((1, batched_data['rgb'].shape[1], 1, 1))
        discretized = torch.unsqueeze(discretized, dim=1).repeat((1, batched_data['depth'].shape[1], 1, 1))

        return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise * 5



class Conv_Controller_AE(nn.Module):
    # Total layer refers to the total available compute budget
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4):
        super(Conv_Controller_AE, self).__init__()
        # Do the respective resizes before the AE function
        self.encoder_dict = nn.ModuleDict({
            'rgb': nn.Sequential(
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
            'depth': nn.Sequential(
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
            'rgb': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(14, 14)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(12, 12), stride=(3, 3)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5)),
            ),
            'depth': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(14, 14)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(12, 12), stride=(3, 3)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5)),
            ),
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        
    def forward(self, image_data = None, depth_data = None):
        conv_embeds = []
        

        b, n, c, h, w = image_data.shape
        out = self.encoder_dict['rgb'](torch.reshape(image_data[:, 0:3], (-1, c, h, w))) # Take only the noise of the first three frames
        conv_embeds.append(torch.reshape(out, (b, 3, -1)))

        b, n, c, h, w = depth_data.shape
        out = self.encoder_dict['depth'](torch.reshape(depth_data[:, 0:3], (-1, c, h, w))) # Take only the noise of the first three frames
        conv_embeds.append(torch.reshape(out, (b, 3, -1)))
        
        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output
        x = torch.reshape(x, (-1, 1, 16, 16))
        
        img_recon = self.decoder_dict['rgb'](x)
        depth_recon = self.decoder_dict['depth'](x)
        return img_recon, depth_recon




# the modality selector for AdaMML
class AdaMML_Modality_Selector(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4):
        super(AdaMML_Modality_Selector, self).__init__()
        
        # Downsample input to 100 x 100 and then pass through conv layers
        self.encoder_dict = nn.ModuleDict({
            'rgb': nn.Sequential(
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
            'depth': nn.Sequential(
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

    def forward(self, batched_data):
        conv_embeds = []

        if 'rgb' in batched_data.keys():
            b, n, c, h, w = batched_data['rgb'].shape
            out = self.encoder_dict['rgb'](torch.reshape(batched_data['rgb'][:, 0:3], (-1, c, h, w))) # Take only the noise of the first frame
            conv_embeds.append(torch.reshape(out, (b, 3, -1)))
        if 'depth' in batched_data.keys():
            b, n, c, h, w = batched_data['depth'].shape
            out = self.encoder_dict['depth'](torch.reshape(batched_data['depth'][:, 0:3], (-1, c, h, w))) # Take only the noise of the first frame
            conv_embeds.append(torch.reshape(out, (b, 3, -1)))
        conv_embeds = torch.cat(conv_embeds, dim=1)
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




if __name__ == '__main__':
    test = torch.rand(64, 24)
    res = softmax_k(test)
    import pdb; pdb.set_trace()
