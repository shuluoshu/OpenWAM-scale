"""Read-only spatial goal memory. The adapter owns every new trainable weight."""
import torch
from torch import nn
from einops import rearrange


class GoalMemoryAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.patch = nn.Linear(48 * 2 * 2, 512)
        self.row_position = nn.Parameter(torch.zeros(1, 8, 1, 512))
        self.column_position = nn.Parameter(torch.zeros(1, 1, 10, 512))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, 512))
        nn.init.normal_(self.row_position, std=0.02)
        nn.init.normal_(self.column_position, std=0.02)
        nn.init.normal_(self.goal_type, std=0.02)
        width = backbone.num_heads * backbone.head_dim
        self.keys = nn.ModuleList(nn.Linear(512, width) for _ in range(backbone.num_layers))
        self.values = nn.ModuleList(nn.Linear(512, width) for _ in range(backbone.num_layers))

    def forward(self, latent, backbone):
        if tuple(latent.shape[1:]) != (48, 1, 16, 20):
            raise ValueError('goal latent must be [B,48,1,16,20]')
        tokens = rearrange(latent.detach(), 'b c 1 (h p) (w q) -> b (h w) (c p q)', p=2, q=2)
        spatial_position = (self.row_position + self.column_position).flatten(1, 2)
        tokens = self.patch(tokens) + spatial_position + self.goal_type
        # Borrow each layer's key norm at call time: never register an alias,
        # and never add queries or a residual goal stream to the joint loop.
        keys = torch.stack([block.self_attn.norm_k(proj(tokens))
                            for block, proj in zip(backbone.dit.blocks, self.keys)])
        values = torch.stack([proj(tokens) for proj in self.values])
        return keys, values


def append_goal_mask(base, token_valid, observation_queries):
    """Preserve even rectangular prefix-KV masks; add only readable goal columns."""
    if token_valid is None or token_valid.dtype != torch.bool or token_valid.ndim != 2 or token_valid.shape[1] != 80:
        raise ValueError('goal token validity must be [B,80]')
    tail = token_valid[:, None, None, :].expand(-1, 1, base.shape[-2], -1).clone()
    tail[:, :, :observation_queries] = False
    return torch.cat((base[None, None].expand(token_valid.shape[0], 1, -1, -1), tail), dim=-1)
