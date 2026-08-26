import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadLinearSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),nn.Dropout(dropout))
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, context)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SharedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            attn_layer = MultiHeadLinearSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                attn_layer,
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        return x

class PromptGuidedEndmemberTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth=2, heads=4, mlp_dim,
                 num_endmembers, dim_head=64, dropout=0., emb_dropout=0.): 
        super().__init__()
        self.dim = dim
        self.P = num_endmembers
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        h_patches = image_height // patch_height
        w_patches = image_width // patch_width
        num_patches = h_patches * w_patches
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_height * patch_width * dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.shared_transformer = SharedTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.prompt_proj = nn.Linear(dim, dim)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.Dropout(0.1) 
            ) for _ in range(self.P)
        ])
        self.to_latent = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=h_patches, w=w_patches))

    def forward(self, img, prompt_embeddings):       
        x = self.to_patch_embedding(img)
        n = x.shape[1]
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        shared_features = self.shared_transformer(x)

        prompts = prompt_embeddings.transpose(0, 1)
        prompts = self.prompt_proj(prompts)

        similarity = torch.matmul(shared_features, prompts.transpose(0, 1))
        similarity = similarity / (self.dim ** 0.5)
        scale = 2.0
        masks = torch.softmax(similarity * scale, dim=-1)
        outputs = []

        for i in range(self.P):
            mask_i = masks[:, :, i:i+1]
            specific_feat = shared_features * mask_i
            specific_feat = self.adapters[i](specific_feat)
            out_map = self.to_latent(specific_feat)
            outputs.append(out_map)

        return outputs




