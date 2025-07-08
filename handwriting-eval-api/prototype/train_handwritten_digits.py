import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

# === 自前データクラス ===
class HandwrittenDigits(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        label = int(img_name.split('_')[0])
        img = Image.open(os.path.join(self.root_dir, img_name)).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label

# === 変換処理 ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === データセット作成 ===
custom_dataset = HandwrittenDigits("result/ml_digits", transform)
mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
combined_dataset = ConcatDataset([custom_dataset, mnist_dataset])
train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# === モデル定義 ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(26*26*16, 128)
        self.fc2 = nn.Linear(128, 11)  # 0〜10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 26*26*16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === 学習処理 ===
model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

torch.save(model.state_dict(), "digit_model.pt")
print("モデルを保存したで！")
