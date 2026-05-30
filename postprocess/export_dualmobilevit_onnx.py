import torch
import torch.nn as nn
import yaml

from models import get_model


class DualMobileViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        branch_a_out, branch_b_out, _ = self.model(x, branch=None)
        return branch_a_out, branch_b_out


def main():
    config_path = "default_joint_config.yaml"
    checkpoint_path = "ckpts/joint.pth"
    output_onnx_path = "dualmobilevit.onnx"

    cfg = yaml.safe_load(open(config_path, "r"))
    model_cfg = cfg["model"]

    model = get_model(model_cfg["model_name"], model_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    model.eval().cuda()

    wrapped_model = DualMobileViTWrapper(model).eval().cuda()

    dummy = torch.randn(1, 3, 224, 224).cuda()

    torch.onnx.export(
        wrapped_model,
        dummy,
        output_onnx_path,
        input_names=["input"],
        output_names=["branch_a_out", "branch_b_out"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"Saved ONNX to {output_onnx_path}")


if __name__ == "__main__":
    main()