import os
import torch
import segmentation_models_pytorch as smp

from model import initialize_model
from gui.layout import build_gradio_ui
from gui.callbacks import register_model

# 1. GPU/Device config
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"

# 2. Load model
MODEL_PATH = "./pi_seg/model/checkpoints/deepPI_lucchi_deploy.pth"
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = initialize_model()
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.to(DEVICE).eval()

# 3. Register to callbacks
register_model(model, DEVICE)

# 4. Launch GUI
CANVAS_SIZE = (640, 640)
POINT_COLORS = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]  # RGB
demo = build_gradio_ui(DEVICE, CANVAS_SIZE, POINT_COLORS)


if __name__ == "__main__":
    demo.queue().launch(share=True)