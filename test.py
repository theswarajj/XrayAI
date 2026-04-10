import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import sys
import os

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "./checkpoints/chexpert_densenet_best.pth"
IMG_SIZE   = 224
THRESHOLD  = 0.5

LABEL_NAMES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
num_classes = checkpoint["model_state"]["classifier.weight"].shape[0]

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print(f"Model loaded  |  classes={num_classes}  |  device={device}")

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Load Image
# ----------------------------
img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

if not os.path.exists(img_path):
    print(f"ERROR: Image not found -> {img_path}")
    sys.exit(1)

image = Image.open(img_path).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)   # (1, C, H, W)

# ----------------------------
# Inference
# ----------------------------
with torch.no_grad():
    logits = model(tensor)                            # (1, num_classes)
    probs  = torch.sigmoid(logits).cpu().numpy()[0]  # (num_classes,)

# ----------------------------
# Results
# ----------------------------
print(f"\n===== PREDICTION: {os.path.basename(img_path)} =====")

if num_classes == 1:
    # Binary mode
    prob = float(probs[0])
    label = "ABNORMAL" if prob >= THRESHOLD else "NORMAL"
    print(f"  Prediction : {label}")
    print(f"  Confidence : {prob:.4f}")
else:
    # Multi-label mode
    labels = LABEL_NAMES[:num_classes]
    detected = []
    for name, p in zip(labels, probs):
        flag = "✓" if p >= THRESHOLD else " "
        print(f"  [{flag}] {name:<35} {p:.4f}")
        if p >= THRESHOLD:
            detected.append(name)
    print(f"\n  Detected conditions: {detected if detected else ['None (Normal)']}")