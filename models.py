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

    def forward(self, x, branch=None):
        shared_features = self.shared_forward(x)

        branch_a_out = None
        branch_b_out = None

        if branch is None or branch == "a":
            branch_a_x = shared_features
            branch_a_x = self.branch_a_stages(branch_a_x)
            branch_a_x = self.branch_a_final_conv(branch_a_x)
            branch_a_x = branch_a_x.mean([-2, -1])
            branch_a_out = self.branch_a_head(branch_a_x)

        if branch is None or branch == "b":
            branch_b_x = shared_features
            branch_b_x = self.branch_b_stages(branch_b_x)
            branch_b_x = self.branch_b_final_conv(branch_b_x)
            branch_b_x = branch_b_x.mean([-2, -1])
            branch_b_out = self.branch_b_head(branch_b_x)

        return branch_a_out, branch_b_out


def get_model(model_name, model_cfg):
    if "mobilevit" in model_name:
        return DualMobileViT(
            model_name=model_name,
            pretrained=model_cfg.get("pretrained", True),
            shared_stages=model_cfg.get("shared_stages", 3),
            branch_a_num_classes=model_cfg.get("branch_a_num_classes", 256),
            branch_b_num_classes=model_cfg.get("branch_b_num_classes", 1),
            head_hidden_dim=model_cfg.get("head_hidden_dim", 256),
            head_drop_rate=model_cfg.get("head_drop_rate", 0.5),
        )

    raise ValueError(f"Unknown model name: {model_name}")
