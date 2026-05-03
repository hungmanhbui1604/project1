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
        shared_blocks=4,
        branch_a_num_classes=128,
        branch_b_num_classes=1,
        head_hidden_dim=256,
        head_drop_rate=0.5,
    ):
        super().__init__()

        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.norm_pre = base_model.norm_pre

        embed_dim = base_model.embed_dim

        self.shared_blocks = base_model.blocks[:shared_blocks]

        # --- branch A ---
        self.branch_a_blocks = base_model.blocks[shared_blocks:]
        self.branch_a_norm = base_model.norm
        self.branch_a_head = nn.Linear(embed_dim, branch_a_num_classes)

        # --- branch B ---
        self.branch_b_blocks = copy.deepcopy(base_model.blocks[shared_blocks:])
        self.branch_b_norm = copy.deepcopy(base_model.norm)
        self.branch_b_head = MLPHead(
            in_features=embed_dim,
            hidden_features=head_hidden_dim,
            out_features=branch_b_num_classes,
            drop_rate=head_drop_rate,
        )

    def shared_forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm_pre(x)
        x = self.shared_blocks(x)
        return x

    def branch_forward(self, x, branch):
        x = self.shared_forward(x)
        if branch == 'a':
            x = self.branch_a_blocks(x)
            x = self.branch_a_norm(x)
            out = self.branch_a_head(x[:, 0].contiguous())
        elif branch == 'b':
            x = self.branch_b_blocks(x)
            x = self.branch_b_norm(x)
            out = self.branch_b_head(x[:, 0].contiguous())
        else:
            raise ValueError(f"Invalid branch: {branch}")
        return out

    def forward(self, x):
        shared_features = self.shared_forward(x)

        branch_a_x = shared_features
        branch_a_x = self.branch_a_blocks(branch_a_x)
        branch_a_norm_x = self.branch_a_norm(branch_a_x)
        branch_a_out = self.branch_a_head(branch_a_norm_x[:, 0].contiguous())

        branch_b_x = shared_features
        branch_b_x = self.branch_b_blocks(branch_b_x)
        branch_b_norm_x = self.branch_b_norm(branch_b_x)
        branch_b_out = self.branch_b_head(branch_b_norm_x[:, 0].contiguous())

        return branch_a_out, branch_b_out


class DualMobileViT(nn.Module):
    def __init__(
        self,
        model_name="mobilevit_s.cvnets_in1k",
        pretrained=True,
        shared_stages=3,
        branch_a_num_classes=128,
        branch_b_num_classes=1,
        head_hidden_dim=256,
        head_drop_rate=0.5,
    ):
        super().__init__()

        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # --- Shared Part ---
        self.shared_stem = base_model.stem
        self.shared_stages = base_model.stages[:shared_stages]
        num_features = base_model.num_features

        # --- Branch A ---
        self.branch_a_stages = base_model.stages[shared_stages:]
        self.branch_a_final_conv = base_model.final_conv
        self.branch_a_head = nn.Linear(num_features, branch_a_num_classes)

        # --- Branch B ---
        self.branch_b_stages = copy.deepcopy(base_model.stages[shared_stages:])
        self.branch_b_final_conv = copy.deepcopy(base_model.final_conv)
        self.branch_b_head = MLPHead(
            in_features=num_features,
            hidden_features=head_hidden_dim,
            out_features=branch_b_num_classes,
            drop_rate=head_drop_rate,
        )

    def shared_forward(self, x):
        x = self.shared_stem(x)
        out = self.shared_stages(x)
        return out

    def branch_forward(self, x, branch):
        x = self.shared_forward(x)
        if branch == 'a':
            x = self.branch_a_stages(x)
            x = self.branch_a_final_conv(x)
            x = x.mean([-2, -1])
            out = self.branch_a_head(x)
        elif branch == 'b':
            x = self.branch_b_stages(x)
            x = self.branch_b_final_conv(x)
            x = x.mean([-2, -1])
            out = self.branch_b_head(x)
        else:
            raise ValueError(f"Invalid branch: {branch}")
        return out
    
    def forward(self, x):
        shared_features = self.shared_forward(x)

        branch_a_x = shared_features
        branch_a_x = self.branch_a_stages(branch_a_x)
        branch_a_x = self.branch_a_final_conv(branch_a_x)
        branch_a_x = branch_a_x.mean([-2, -1])
        branch_a_out = self.branch_a_head(branch_a_x)

        branch_b_x = shared_features
        branch_b_x = self.branch_b_stages(branch_b_x)
        branch_b_x = self.branch_b_final_conv(branch_b_x)
        branch_b_x = branch_b_x.mean([-2, -1])
        branch_b_out = self.branch_b_head(branch_b_x)

        return branch_a_out, branch_b_out


def get_model(model_name, model_cfg):
    if "mobilevit_" in model_name:
        return DualMobileViT(
            model_name=model_name,
            pretrained=model_cfg.get("pretrained", True),
            shared_stages=model_cfg.get("shared_stages", 3),
            branch_a_num_classes=model_cfg.get("branch_a_num_classes", 256),
            branch_b_num_classes=model_cfg.get("branch_b_num_classes", 1),
            head_hidden_dim=model_cfg.get("head_hidden_dim", 256),
            head_drop_rate=model_cfg.get("head_drop_rate", 0.5),
        )
    elif "vit_" in model_name:
        return DualViT(
            model_name=model_name,
            pretrained=model_cfg.get("pretrained", True),
            shared_blocks=model_cfg.get("shared_blocks", 4),
            branch_a_num_classes=model_cfg.get("branch_a_num_classes", 256),
            branch_b_num_classes=model_cfg.get("branch_b_num_classes", 1),
            head_hidden_dim=model_cfg.get("head_hidden_dim", 256),
            head_drop_rate=model_cfg.get("head_drop_rate", 0.5),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
