from multiprocessing import resource_sharer
import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

# helpers
# TODO figure out prenorm

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.cuda()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.condln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Identity(),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.condln(x)
        return self.net(x)

class OrdinaryFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class OrdinarySelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mem):
        x = self.norm(x)

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        kv = self.to_kv(mem).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class OrdinaryCrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mem):
        x = self.norm(x)

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        kv = self.to_kv(mem).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# TODO LayerNorm is learnable, we should probably place one layernorm in each block
# 2 * num layernorms + 1 gammas and betas
class TransformerEnc(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
            ]))

    def forward(self, x):
        # 2 layernorm per layer
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# Normal Layernorm
class OrdinaryTransformerEnc(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                OrdinarySelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                OrdinaryFeedForward(dim, mlp_dim, dropout = dropout),
            ]))

    def forward(self, x):
        # 2 layernorm per layer
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TransformerDec(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mem):
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, mem) + x
            x = ff(x) + x
        return self.norm(x)

class transformMLP(nn.Module): # Not used for the time being
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(7, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 9)
    )

  def forward(self, x):
    return self.layers(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean', "all"}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEnc(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # img = rearrange(img, "b h w c -> b c h w") # bs, # of feature vectors (288 for image, 100 for depth, 64 for mmWave), feature dims(e.g. 128/256)
        x = self.to_patch_embedding(img)
        #import ipdb; ipdb.set_trace()
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # bs x 1 x feature dim
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
       
        x = self.transformer(x)
        if self.pool== "all":
            return x[:,1:]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # return self.mlp_head(x)
        #import ipdb; ipdb.set_trace()
        
        return x 

class LIMUBertEnc(nn.Module):
    def __init__(self, sig_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 4, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        num_win, win_len = sig_size
        patch_dim = channels * win_len
        self.projector = nn.Sequential(
            Rearrange('b n l c -> b n (l c)'), # bs x num_windows x window_length x channels
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_win + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerEnc(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
    
    def forward(self, sig):
        x = self.projector(sig)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # bs x 1 x feature dim
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        if self.pool== "all":
            return x[:,1:]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return x


# class Celestracker(nn.Module): # Not used for the time being
#     def __init__(self, dim, dim_vit, depth_vit, depth_E, depth_D, heads, mlp_dim, n_cls = 1, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0., n_class=3):
#         self.dim_vit = dim_vit
#         self.obj_query = nn.Parameter(torch.randn(1, 10, dim))
#         self.ViT = ViT(image_size=(270, 480), patch_size=(15, 30), dim=3*dim_vit, depth=depth_vit, heads=heads, 
#                        mlp_dim=3*dim_vit, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.)
#         self.transformMLP = transformMLP()
#         self.DETREnc = TransformerEnc(dim=dim, depth=depth_E, heads=heads, dim_head=dim_head, mlp_dim=3*dim, dropout = 0.)
#         self.DETRDec = TransformerDec(dim=dim, depth=depth_D, heads=heads, dim_head=dim_head, mlp_dim=3*dim, dropout = 0.)
#         self.cls_output = nn.Linear(dim, n_class+1) # class of queries, include 0
#         self.pos_output = nn.Linear(dim, 2) # x,y coordinate of queries

#     def forward(self, img, sensor_pos):
#         batch_size = len(img)
#         img = rearrage(img, 'b n w h -> (b n w h)')
#         trans = self.transformMLP(sensor_pos) # learn a 3x3 transform for features
#         trans = rearrange(trans, 'b (h w) -> b h w', h = 3)
#         feat = self.ViT(img) # extract dim_vit x 3 feature vector from each input vector
#         feat = rearrage(feat, 'b (d w) -> b d w', d=self.dim_vit)
#         feat = torch.matmul(feat, trans)
#         feat = rearrage(feat, '(b n) d w -> b (n w) d', b=batch_size)
#         x = self.DETREnc(feat)
#         x = self.DETRDec(self.obj_query, x)
#         return (self.cls_output(x), self.pos_output(x))
def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return 


if __name__ == '__main__':
    print("Hello World!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # simulated target signal
    cam_img = torch.rand(64,3,270,480).to(device)
    depth_img = torch.rand(64,1,120,160).to(device)
    mmwave_img = torch.rand(64,1,256,16).to(device)
    audio_sig = torch.rand(64, 4, 1056).to(device).cpu().numpy()
    audio_sig_windowed = np.array([np.squeeze(audio_sig[:,:,i:i+88]) for i in range(0,1056-88+1,44)]) # 5ms window x 50% overlap
    audio_sig_windowed = torch.tensor(rearrange(audio_sig_windowed, "w b c l -> b w l c")).to(device) # bs x 23 x 88 x 4 
    

    dim_vit = 128
    depth_vit = 3
    heads = 4
    dropout = 0.2
    emb_dropout = 0.2
    img_backbone = ViT(image_size=(270, 480), patch_size=(15, 30), dim=dim_vit, depth=depth_vit, heads=heads, 
                       mlp_dim=3*dim_vit, pool = 'cls', channels = 3, dim_head = dim_vit//heads, dropout = 0., emb_dropout = 0.).to(device)
    count_parameters(img_backbone)
    depth_backbone = ViT(image_size=(120, 160), patch_size=(12, 16), dim=dim_vit, depth=depth_vit, heads=heads, 
                       mlp_dim=3*dim_vit, pool = 'cls', channels = 3, dim_head = dim_vit//heads, dropout = 0., emb_dropout = 0.).to(device)
    count_parameters(depth_backbone)
    mmwave_backbone = ViT(image_size=(256, 16), patch_size=(16, 4), dim=dim_vit, depth=depth_vit, heads=heads, 
                       mlp_dim=3*dim_vit, pool = 'cls', channels = 1, dim_head = dim_vit//heads, dropout = 0., emb_dropout = 0.).to(device)
    count_parameters(mmwave_backbone)
    audio_backbone = LIMUBertEnc(sig_size=(64,88), dim=dim_vit, depth=depth_vit, heads=heads, 
                       mlp_dim=3*dim_vit, pool = 'cls', channels = 4, dim_head = dim_vit//heads, dropout = 0., emb_dropout = 0.).to(device)
    count_parameters(audio_backbone)
    cam_output = img_backbone(cam_img)
    depth_output = depth_backbone(depth_img)
    mmwave_output = mmwave_backbone(mmwave_img)
    audio_output = audio_backbone(audio_sig_windowed)
    import pdb; pdb.set_trace()
    
    


