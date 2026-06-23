from flask import Flask, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convi1 = nn.Conv2d(3, 96, 11, 4) # Input
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LocalResponseNorm(5, 0.0001, 0.75, 2)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.convi2 = nn.Conv2d(96, 256, 5, 1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.LocalResponseNorm(5, 0.0001, 0.75, 2)
        self.pool2 = nn.MaxPool2d(3, 2)

        self.convi3 = nn.Conv2d(256, 384, 3, 1)
        self.relu3 = nn.ReLU()

        self.convi4 = nn.Conv2d(384, 384, 3, 1)
        self.relu4 = nn.ReLU()

        self.convi5 = nn.Conv2d(384, 256, 3, 1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(3, 2)

        self.flatten = nn.Flatten()
        self.fc6 = nn.Linear(256, 4096) #Change this for actual values
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.convi1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.convi2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.convi3(x)
        x = self.relu3(x)
        x = self.convi4(x)
        x = self.relu4(x)
        x = self.convi5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x
app = Flask(__name__)

model = cnn()
model.eval()

x = torch.randn(1, 3, 224, 224) #Image, randomn init rn
out = model(x)
print(torch.xpu.is_available()) #Need to add support for intel gpu
print(torch.xpu.device_count())
print(out.shape)
@app.route("/")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)