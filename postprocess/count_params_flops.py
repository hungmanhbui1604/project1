import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import yaml
from models import get_model
from thop import profile, clever_format

cfg = yaml.safe_load(open("default_joint_config.yaml", "r"))
model_cfg = cfg["model"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(model_cfg["model_name"], model_cfg)
model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Params: {total_params / 1e6:.2f} M")
print(f"Trainable Params: {trainable_params / 1e6:.2f} M")

dummy = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    macs, _ = profile(model, inputs=(dummy,), verbose=False)

macs_fmt = clever_format(macs, "%.2f")
flops_fmt = clever_format(2 * macs, "%.2f")

print(f"Formatted MACs: {macs_fmt}")
print(f"Formatted FLOPs: {flops_fmt}")