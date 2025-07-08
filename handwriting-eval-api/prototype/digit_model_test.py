import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# モデルロード
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1)
        self.fc1 = torch.nn.Linear(26*26*16, 128)
        self.fc2 = torch.nn.Linear(128, 11)  # 0〜10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 26*26*16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 画像から数字を推論する関数
def predicted_image(img):
    # 背景白・文字黒なら反転（MNISTに合わせる）
    img = ImageOps.invert(img)

    # 必要なら余白を自動クロップして28x28にフィット
    img = ImageOps.crop(img)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    input_tensor = transform(img).unsqueeze(0)

    # 推論
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        #print(f'Predicted digit: {predicted.item()}')
    return predicted.item()

model = SimpleCNN()
model.load_state_dict(torch.load("/workspace/data/digit_model.pt"))
model.eval()

# テスト画像
for i in range(1, 13):
    img = Image.open(f'/workspace/result/scores/score_{i}.png').convert('L')
    predicted_digit = predicted_image(img)
    print(f'Predicted digit {i}: {predicted_digit}')

