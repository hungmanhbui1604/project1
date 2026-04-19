import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility Modules
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Stochastic depth (drop path) for residual connections."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class Mlp(nn.Module):
    """2-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Window Helpers
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: window size M

    Returns:
        windows: (num_windows * B, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.

    Args:
        windows: (num_windows * B, M, M, C)
        window_size: window size M
        H, W: original spatial dimensions

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Core Swin Transformer Components
# ---------------------------------------------------------------------------


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.

    Supports both regular (W-MSA) and shifted (SW-MSA) window attention
    via an optional attention mask.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Relative position bias table: (2M-1)*(2M-1) entries, one per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute relative position index for each token pair in a window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, M, M)
        coords_flatten = torch.flatten(coords, 1)  # (2, M*M)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, M*M, M*M)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (M*M, M*M, 2)
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (M*M, M*M)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows * B, M*M, C)
            mask: (num_windows, M*M, M*M) or None

        Returns:
            x: (num_windows * B, M*M, C)
        """
        B_, N, C = x.shape  # B_ = num_windows * B, N = M*M

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # broadcast over batch

        # Apply attention mask for shifted windows
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # (1, nW, 1, N, N)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    A single Swin Transformer block.

    Alternates between regular window attention (W-MSA, shift_size=0) and
    shifted window attention (SW-MSA, shift_size=window_size//2).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=drop,
        )

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Compute attention mask for shifted window attention."""
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (nW, M, M, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # (nW, M*M)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, M*M, M*M)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions

        Returns:
            x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, M, M, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, M*M, C)

        # Compute attention mask
        attn_mask = self._compute_attn_mask(H, W, x.device)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (nW*B, M*M, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual connections
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ---------------------------------------------------------------------------
# Patch Embedding & Merging
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """Split image into 4×4 patches and project to embedding dimension."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            x: (B, Hp*Wp, embed_dim)
            Hp, Wp: spatial dimensions after patching
        """
        x = self.proj(x)  # (B, embed_dim, H/4, W/4)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, embed_dim)
        x = self.norm(x)
        return x, Hp, Wp


class PatchMerging(nn.Module):
    """
    Patch merging layer for spatial downsampling between stages.

    Concatenates features from 2×2 neighboring patches (4C channels),
    then applies LayerNorm and a linear projection to 2C channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions

        Returns:
            x: (B, H/2 * W/2, 2C)
            H_new, W_new: new spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        # Merge 2×2 neighboring patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        H_new, W_new = H // 2, W // 2
        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2C)
        return x, H_new, W_new


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class SwinTransformerStage(nn.Module):
    """
    A Swin Transformer stage: optional PatchMerging followed by a sequence of
    SwinTransformerBlocks with alternating W-MSA and SW-MSA.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: list = None,
        downsample: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Patch merging for downsampling (input dim is dim//2 before merging)
        if downsample:
            self.downsample = PatchMerging(dim // 2)
        else:
            self.downsample = None

        # Build blocks with alternating W-MSA (shift=0) and SW-MSA (shift=window_size//2)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if drop_path is not None else 0.0,
                )
            )

    def forward(self, x: torch.Tensor, H: int, W: int):
        """
        Args:
            x: (B, H*W, C)
            H, W: current spatial dimensions

        Returns:
            x: (B, H_new*W_new, C_new)
            H_new, W_new: updated spatial dimensions
        """
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        for block in self.blocks:
            x = block(x, H, W)

        return x, H, W


# ---------------------------------------------------------------------------
# Full Models
# ---------------------------------------------------------------------------


class SwinTransformerTiny(nn.Module):
    """
    Swin Transformer Tiny.

    Architecture:
        PatchEmbedding → Stage 1 (2 blocks, 96ch) → Stage 2 (2 blocks, 192ch) →
        Stage 3 (6 blocks, 384ch) → Stage 4 (2 blocks, 768ch) →
        LayerNorm → AdaptiveAvgPool → Linear → embedding
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 512,
        swin_embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.swin_embed_dim = swin_embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, swin_embed_dim, patch_size)

        # Stochastic depth: linearly increasing drop rates across all blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Build 4 stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage_dim = swin_embed_dim * (2 ** i)
            stage_drop_path = dpr[sum(depths[:i]):sum(depths[:i + 1])]
            self.stages.append(
                SwinTransformerStage(
                    dim=stage_dim,
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=stage_drop_path,
                    downsample=(i > 0),
                )
            )

        # Final norm and embedding head
        final_dim = swin_embed_dim * (2 ** (self.num_stages - 1))  # 768 for Swin-T
        self.norm = nn.LayerNorm(final_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(final_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            embedding: (B, embed_dim)
        """
        x, H, W = self.patch_embed(x)

        for stage in self.stages:
            x, H, W = stage(x, H, W)

        x = self.norm(x)  # (B, H*W, C)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)  # (B, C)
        x = self.head(x)  # (B, embed_dim)
        return x


class DualSwinTransformerTiny(nn.Module):
    """
    Dual-branch Swin Transformer Tiny.

    Shared:  PatchEmbedding + Stage 1 + Stage 2
    Branch A: Stage 3 + Stage 4 + Head → emb_a  (B, embed_dim_a)
    Branch B: Stage 3 + Stage 4 + Head → emb_b  (B, embed_dim_b)
    Output:  (emb_a, emb_b)

    Architecture diagram:

        Input Image (224×224)
             │
        ┌────┴────────────────────┐
        │  Shared PatchEmbedding  │  96ch, H/4 × W/4
        │  Shared Stage 1        │  2 blocks, 96ch
        │  Shared Stage 2        │  2 blocks, 192ch
        └────────────┬───────────┘
               ┌─────┴─────┐
               ▼           ▼
          Branch A     Branch B
          Stage 3      Stage 3     (6 blocks, 384ch)
          Stage 4      Stage 4     (2 blocks, 768ch)
          Norm+Pool    Norm+Pool
          FC→emb_a     FC→emb_b
               │           │
               ▼           ▼
           (B, dim_a)  (B, dim_b)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim_a: int = 512,
        embed_dim_b: int = 512,
        swin_embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.swin_embed_dim = swin_embed_dim

        # Stochastic depth rates across all 12 blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # ── Shared layers ────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(in_channels, swin_embed_dim, patch_size)

        # Stage 1: 96ch, 2 blocks, no downsampling
        self.shared_stage1 = SwinTransformerStage(
            dim=swin_embed_dim,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[:depths[0]],
            downsample=False,
        )

        # Stage 2: 192ch, 2 blocks, with PatchMerging (96→192)
        self.shared_stage2 = SwinTransformerStage(
            dim=swin_embed_dim * 2,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[depths[0]:depths[0] + depths[1]],
            downsample=True,
        )

        # ── Branch A ─────────────────────────────────────────────────────
        branch_offset = depths[0] + depths[1]

        # Stage 3: 384ch, 6 blocks, with PatchMerging (192→384)
        self.branch_a_stage3 = SwinTransformerStage(
            dim=swin_embed_dim * 4,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[branch_offset:branch_offset + depths[2]],
            downsample=True,
        )

        # Stage 4: 768ch, 2 blocks, with PatchMerging (384→768)
        self.branch_a_stage4 = SwinTransformerStage(
            dim=swin_embed_dim * 8,
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[branch_offset + depths[2]:],
            downsample=True,
        )

        final_dim = swin_embed_dim * 8  # 768
        self.branch_a_norm = nn.LayerNorm(final_dim)
        self.branch_a_pool = nn.AdaptiveAvgPool1d(1)
        self.branch_a_head = nn.Linear(final_dim, embed_dim_a)

        # ── Branch B ─────────────────────────────────────────────────────
        self.branch_b_stage3 = SwinTransformerStage(
            dim=swin_embed_dim * 4,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[branch_offset:branch_offset + depths[2]],
            downsample=True,
        )

        self.branch_b_stage4 = SwinTransformerStage(
            dim=swin_embed_dim * 8,
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[branch_offset + depths[2]:],
            downsample=True,
        )

        self.branch_b_norm = nn.LayerNorm(final_dim)
        self.branch_b_pool = nn.AdaptiveAvgPool1d(1)
        self.branch_b_head = nn.Linear(final_dim, embed_dim_b)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _branch_forward(self, x, H, W, stage3, stage4, norm, pool, head):
        """Run a single branch (stages 3-4 + head)."""
        x, H, W = stage3(x, H, W)
        x, H, W = stage4(x, H, W)
        x = norm(x)
        x = pool(x.transpose(1, 2)).flatten(1)
        x = head(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            emb_a: (B, embed_dim_a)
            emb_b: (B, embed_dim_b)
        """
        # Shared
        x, H, W = self.patch_embed(x)
        x, H, W = self.shared_stage1(x, H, W)
        x, H, W = self.shared_stage2(x, H, W)

        # Branch A
        emb_a = self._branch_forward(
            x, H, W,
            self.branch_a_stage3, self.branch_a_stage4,
            self.branch_a_norm, self.branch_a_pool, self.branch_a_head,
        )

        # Branch B
        emb_b = self._branch_forward(
            x, H, W,
            self.branch_b_stage3, self.branch_b_stage4,
            self.branch_b_norm, self.branch_b_pool, self.branch_b_head,
        )

        return emb_a, emb_b
