import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import RAFDBDataset
from model import build_model
from loss import FocalLoss
from utils import evaluate
from early_stopping import EarlyStopping

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 35
LR = 1e-4
IMG_SIZE = 224

# =========================
# GPU INFO
# =========================
def print_gpu_info():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        name = torch.cuda.get_device_name(gpu_id)
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3

        print("✅ CUDA AVAILABLE")
        print(f"🖥️ GPU: {name}")
        print(f"💾 Total VRAM: {total_mem:.2f} GB")
    else:
        print("❌ CUDA NOT AVAILABLE — USING CPU")

print_gpu_info()

# =========================
# TRANSFORMS
# =========================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS & LOADERS
# =========================
train_ds = RAFDBDataset(
    root_dir="dataset/train",
    csv_file="dataset/train_labels.csv",
    transform=train_tfms
)

test_ds = RAFDBDataset(
    root_dir="dataset/test",
    csv_file="dataset/test_labels.csv",
    transform=val_tfms
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,        # 🔴 CHANGE HERE
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,        # 🔴 CHANGE HERE
    pin_memory=True
)


# =========================
# MODEL, LOSS, OPTIMIZER
# =========================
model = build_model(num_classes=7).to(DEVICE)

criterion = FocalLoss(gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

early_stopper = EarlyStopping(patience=6)
best_acc = 0.0

# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc, cm = evaluate(model, test_loader, DEVICE)


    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"📉 Train Loss (avg): {avg_loss:.4f}")
    print(f"🎯 Test Accuracy: {acc:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_efficientnet.pth")
        print("💾 Best model saved")

    # Early stopping
    if early_stopper.step(acc):
        print("⏹️ Early stopping triggered")
        break

print("\n✅ Training complete")
print(f"🏆 Best Accuracy: {best_acc:.4f}")
