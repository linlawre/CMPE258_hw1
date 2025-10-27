import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from torchvision.models import ResNet18_Weights

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./dataset"  # folder containing bird/, cat/, dog/
BATCH_SIZE = 32
EPOCHS = 5

# -------------------------
# Dataset and Dataloader
# -------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -------------------------
# Define Models
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def get_resnet18(num_classes=3):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # adjust for 64x64
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_mobilenetv2(num_classes=3):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# -------------------------
# Training Functions
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def train_model(model, loader, criterion, optimizer, epochs=5):
    history = {"loss": [], "acc": []}
    model.to(DEVICE)
    for ep in range(epochs):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer)
        print(f"Epoch {ep+1}/{epochs} - Loss: {loss:.4f}, Acc: {acc:.4f}")
        history["loss"].append(loss)
        history["acc"].append(acc)
    return model, history

# -------------------------
# Inference / Speed Test
# -------------------------
def measure_speed(model, name, device=torch.device("cpu")):
    model.eval()
    model.to(device)
    dummy = torch.randn(1, 3, 64, 64).to(device)

    # Warmup
    for _ in range(10): model(dummy)

    # Timing
    t0 = time.time()
    for _ in range(100):
        _ = model(dummy)
    t1 = time.time()
    avg_time = (t1 - t0)/100
    print(f"{name} avg time: {avg_time:.6f}s")
    return avg_time

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Define models
    models_dict = {
        "SimpleCNN": SimpleCNN(num_classes=len(dataset.classes)),
        "ResNet18": get_resnet18(num_classes=len(dataset.classes)),
        "MobileNetV2": get_mobilenetv2(num_classes=len(dataset.classes))
    }

    criterion = nn.CrossEntropyLoss()
    histories = {}
    trained_models = {}

    # -------------------------
    # Training
    # -------------------------
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trained_model, history = train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS)
        histories[name] = history
        trained_models[name] = trained_model

    # -------------------------
    # Plot training comparison
    # -------------------------
    plt.figure(figsize=(12,5))
    for name, history in histories.items():
        plt.plot(history["acc"], label=f"{name} Accuracy")
    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    for name, history in histories.items():
        plt.plot(history["loss"], label=f"{name} Loss")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # -------------------------
    # Inference comparison
    # -------------------------
    for name, model in trained_models.items():
        print(f"\nInference Speed for {name}...")

        # Original PyTorch
        pytorch_time = measure_speed(model, f"{name} Original PyTorch", device=DEVICE)

        # TorchScript
        ts_model = torch.jit.trace(model.cpu(), torch.randn(1,3,64,64))
        ts_time = measure_speed(ts_model, f"{name} TorchScript (CPU)", device=torch.device("cpu"))

        # Dynamic Quantization (CPU only)
        q_model = torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
        q_time = measure_speed(q_model, f"{name} Quantized (CPU)", device=torch.device("cpu"))

        # Plot inference times for this model
        plt.figure()
        plt.bar(["PyTorch", "TorchScript", "Quantized"], [pytorch_time, ts_time, q_time])
        plt.ylabel("Avg Inference Time (s)")
        plt.title(f"Inference Optimization Comparison - {name}")
        plt.show()
