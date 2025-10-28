import torch
from transformers import SiglipImageProcessor, SiglipVisionModel
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load Vision Encoder from a Vision-Language Model
print("Loading Vision Encoder (SigLIP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(device)
model.eval()

# 2. Dataset: your folder structure -> dataset/{bird, cat, dog}
dataset_dir = "dataset"
classes = os.listdir(dataset_dir)

X, y = [], []

print("Encoding images...")
for label, cls in enumerate(classes):
    folder = os.path.join(dataset_dir, cls)
    for file in tqdm(os.listdir(folder), desc=f"Processing {cls}"):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, file)
            image = Image.open(path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy().flatten()

            X.append(embedding)
            y.append(label)

X = np.array(X)
y = np.array(y)

# 3. Train simple classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"✅ Vision Encoder (VLM) Accuracy: {acc:.4f}")
