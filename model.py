import copy

import timm
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop_rate=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm = nn.LayerNorm(hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DualViT(nn.Module):
    def __init__(
        self,
        model_name="vit_small_patch16_224",
        pretrained=True,
        branch_a_num_classes=128,
        branch_b_num_classes=2,
        head_hidden_dim=256,
        head_drop_rate=0.5,
    ):
        super().__init__()

        base_model = timm.create_model(model_name, pretrained=pretrained)

        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.norm_pre = base_model.norm_pre

        embed_dim = base_model.embed_dim

        self.shared_blocks = base_model.blocks[:4]

        # --- branch A ---
        self.branch_a_blocks = base_model.blocks[4:]
        self.branch_a_norm = base_model.norm
        self.branch_a_head = nn.Linear(embed_dim, branch_a_num_classes)

        # --- branch B ---
        self.branch_b_blocks = copy.deepcopy(base_model.blocks[4:8])
        self.branch_b_norm = copy.deepcopy(base_model.norm)
        self.branch_b_head = MLPHead(
            in_features=embed_dim,
            hidden_features=head_hidden_dim,
            out_features=branch_b_num_classes,
            drop_rate=head_drop_rate,
        )

    def forward_features(self, x):
        # embeddings
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm_pre(x)

        # shared blocks
        x = self.shared_blocks(x)
        return x

    def forward(self, x, return_features=False, branch=None):
        # embeddings
        shared_features = self.forward_features(x)

        features = {}
        if return_features:
            features['layer4'] = shared_features

        branch_a_out = None
        if branch is None or branch == 'a':
            branch_a_x = shared_features
            for i, blk in enumerate(self.branch_a_blocks):
                branch_a_x = blk(branch_a_x)
                if return_features and i == 3:
                    features['a_layer8'] = branch_a_x
            if return_features:
                features['a_layer12'] = branch_a_x

            branch_a_norm_x = self.branch_a_norm(branch_a_x)
            branch_a_out = self.branch_a_head(branch_a_norm_x[:, 0].contiguous())

        branch_b_out = None
        if branch is None or branch == 'b':
            branch_b_x = shared_features
            for i, blk in enumerate(self.branch_b_blocks):
                branch_b_x = blk(branch_b_x)
            if return_features:
                features['b_layer8'] = branch_b_x

            branch_b_norm_x = self.branch_b_norm(branch_b_x)
            branch_b_out = self.branch_b_head(branch_b_norm_x[:, 0].contiguous())

        if return_features:
            return branch_a_out, branch_b_out, features

        return branch_a_out, branch_b_out