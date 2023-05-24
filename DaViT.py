from copy import deepcopy
import itertools
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information.
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=k, stride=1, padding=k // 2, groups=dim)

    def forward(self, x, size): # b (hw) c
        H, W = size
        feat = rearrange(x, 'b (h w) c -> b c h w', h=H)
        feat = self.proj(feat)
        pos = rearrange(feat, 'b c h w -> b (h w) c')
        return x + pos


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, overlapped=False):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        if patch_size == 4:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=7,
                                  stride=patch_size,padding=3)
            self.norm = nn.LayerNorm(embed_dim)
        
        if patch_size == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=2,
                                  stride=patch_size, padding=0)
            self.norm = nn.LayerNorm(in_chans)

    def forward(self, x, size):
        H, W = size
        dim = len(x.shape)
        if dim == 3:
            x = self.norm(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H)

        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size- H % self.patch_size))

        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = rearrange(x, 'b c h w -> b (h w) c')
        if dim == 4:
            x = self.norm(x)
        return x, newsize


class ChannelAttention(nn.Module):
    r""" Channel based self attention.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of the groups.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.qkv(x) # B N 3C
        q, k, v = tuple(rearrange(qkv, 'b n (d k h) -> k b h n d ', k=3, h=self.num_heads)) #B N 3C -> 3 B H N C/H 
        scaled_dot_prod = einsum('b h n i , b h n j -> b h i j', k, v) * self.scale # B H C/H N @ B H N C/H -> B H C/H C/H
        attention = scaled_dot_prod.softmax(dim=-1)
        x = einsum('b h i j , b h n j -> b h i n', attention, q) # B H C/H C/H @  B H C/H N  -> B H C/H N  
        x = rearrange(x, "b h d n -> b n (h d)") # B H C/H N -> B N C                                                        
        x = self.linear(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
    """
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.qkv(x)
        #B N 3C -> 3 B H N C/H 
        q, k, v = tuple(rearrange(qkv, 'b n (d k h) -> k b h n d ', k=3, h=self.num_heads))
        scaled_dot_prod = einsum('b h i d , b h j d -> b h i j', q, k) * self.scale # B H N C/H @ B H C/H N -> B H N N
        attention = scaled_dot_prod.softmax(dim=-1)
        x = einsum('b h i j , b h j d -> b h i d', attention, v) # B H N N @ B H N C/H -> B H N C/H
        x = rearrange(x, "b h n d -> b n (h d)") # B H N C/H -> B N C
        x = self.linear(x)
        return x
    
    
class ChannelBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., ffn=True):
        super().__init__()

        self.conv_pose_enc = nn.ModuleList([ConvPosEnc(dim=dim, k=3), ConvPosEnc(dim=dim, k=3)])
        self.ffn = ffn
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = Mlp( in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, size):
        
        x = self.conv_pose_enc[0](x, size) # b (h w) c
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.conv_pose_enc[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size



class SpatialBlock(nn.Module):
    r""" Spatial-wise Local Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        ffn (bool): If False, pure attention network without FFNs
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., drop_path=0.,ffn=True):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.window_size = window_size
        self.conv_pose_enc = nn.ModuleList([ConvPosEnc(dim=dim, k=3), ConvPosEnc(dim=dim, k=3)])

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = self.conv_pose_enc[0](x, size) #b (h w) c
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)
        
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        B, Hp , Wp, C = x.shape
        
        x_windows = rearrange(x, 'b (w_h nw_h) (w_w nw_w) c -> (b nw_h nw_w) (w_h w_w) c', 
                              w_h=self.window_size, w_w=self.window_size) # window_partition
        attn_windows = self.attn(x_windows)      # (b nw_h nw_w) (w_h w_w) c
        
        x = rearrange(attn_windows, '(b nw_h nw_w) (w_h w_w) c -> b (w_h nw_h) (w_w nw_w) c', 
                      b=B, w_h=self.window_size, nw_h=int(Hp / self.window_size)) #window_reverse
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = rearrange(x, ' b h w c -> b (h w) c') 
        x = shortcut + self.drop_path(x)
        x = self.conv_pose_enc[1](x, size)
        
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


class StageModule(nn.Module):
    def __init__(self, dpr_id, in_chans=3, depth=1, patch_size=4, dpr=None,
                 embed_dim=96 , num_head=3, window_size=7, mlp_ratio=2., 
                 ffn=True, overlapped_patch=False):
        super(StageModule, self).__init__()
        
        self.patch_embed = PatchEmbed(patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,
                                      overlapped=overlapped_patch)

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(nn.Sequential(
                                SpatialBlock(dim=embed_dim, num_heads=num_head,
                                    mlp_ratio=mlp_ratio, drop_path=dpr[2*(idx)+dpr_id], 
                                    ffn=ffn, window_size=window_size),
                                ChannelBlock(dim=embed_dim, num_heads=num_head,
                                    mlp_ratio=mlp_ratio, drop_path=dpr[2*(idx)+dpr_id + 1], ffn=ffn)))
        
    def forward(self, x, size):
        x, size = self.patch_embed(x, size) #B HW C 
        for spatial_block, channel_block in self.layers:
            x, _ = spatial_block(x, size)
            x, _ = channel_block(x, size)
        return x, size
        



class DaViT(nn.Module):
    """ Dual-Attention ViT
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        ffn (bool): If False, pure attention network without FFNs
        overlapped_patch (bool): If True, use overlapped patch division during patch merging.
    """
    def __init__(self, in_chans=3, num_classes=1, depths=(1, 1, 9, 1), patch_size=4,
                 embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 drop_path_rate=0.1, ffn=True, overlapped_patch=False):
        super().__init__()
        
        assert len(embed_dims) == len(num_heads) 
        
        self.architecture = [[index] * item for index, item in enumerate(depths)]
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*self.architecture))))] # 24개로 0 ~ 0.1 까지 구간 분할 
        dpr_ids = [len(list(itertools.chain(*self.architecture[:i])))  for i in range(len(embed_dims))]
        
        self.stage1 = StageModule(dpr_id=dpr_ids[0], in_chans=3, depth=depths[0], 
                                  patch_size=4, dpr=dpr, embed_dim=embed_dims[0] , num_head=num_heads[0])
        self.stage2 = StageModule(dpr_id=dpr_ids[1], in_chans=embed_dims[0], depth=depths[1], 
                                  patch_size=2, dpr=dpr, embed_dim=embed_dims[1] , num_head=num_heads[1])
        self.stage3 = StageModule(dpr_id=dpr_ids[2], in_chans=embed_dims[1], depth=depths[2], 
                                  patch_size=2, dpr=dpr, embed_dim=embed_dims[2] , num_head=num_heads[2])
        self.stage4 = StageModule(dpr_id=dpr_ids[3], in_chans=embed_dims[2], depth=depths[3], 
                                  patch_size=2, dpr=dpr, embed_dim=embed_dims[3] , num_head=num_heads[3])
        
        self.norm = nn.LayerNorm(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.embed_dims[-1], num_classes)

    def forward(self, x):
        # B 3 224 224
        x, size = self.stage1(x, (x.size(2), x.size(3)))# B 3136 96
        x, size = self.stage2(x, size) # B 784 192    
        x, size = self.stage3(x, size) # B 196 384
        x, size = self.stage4(x, size) # B 49 768
        x = self.avgpool(x.transpose(1, 2)) # B 768 1
        x = self.norm(torch.flatten(x, 1)) # B 768
        x = self.head(x) # B 1
        return x


model = DaViT(embed_dims=(96, 192, 384, 768), depths=(1, 1, 9, 1))
input = torch.rand(1, 3, 128, 128)
out = model(input)
print(out.shape)

def c(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(c(model))
