from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms import v2

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
        self.fc8 = nn.Linear(4096, 5)
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

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

def runTraining(model, transform, epochCount = 1):
    model.train()
    trainingSet = torchvision.datasets.ImageFolder("animals/train", transform)
    trainingLoader = torch.utils.data.DataLoader(trainingSet, 32, True, num_workers=4, pin_memory=True)
    lossFunc = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), 0.005, 0.9)
    print(epochCount)
    print(f'Training set has {len(trainingSet)} instances')
    for i in range(epochCount):
        lastLoss = 0.0
        for j, data in enumerate(trainingLoader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = lossFunc(outputs, labels)
            loss.backward() # Runs backgrop for gradients
            optimiser.step() #Applies gradients
            lastLoss = loss.item()
            if j % 100 == 0:
                print(j)
        print("Loss from epoch ", i, " = ", lastLoss)
    torch.save(model.state_dict(), "cnn.pth")

def runTest(model, image):
    model.eval()
    outputs = model(image)
    print(outputs)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/train", methods=['GET', 'POST'])
def train():
    if request.method == "POST":
        action = request.form.get("action")
        epochCount = int(request.form.get("epochCount"))
        if action == "beginTrain":
            runTraining(model, transform, epochCount)
    return render_template("train.html")

@app.route("/test", methods=['GET', 'POST'])
def test():
    if request.method == "POST":
        testImage = request.files.get("testImage")
        if testImage is None:
            print("No test image")
        else:
            testImage = Image.open(testImage.stream).convert("RGB")
            testImage = transform(testImage)
            testImage = testImage.unsqueeze(0).to(device)
            runTest(model, testImage)
    return render_template("test.html")

if __name__ == "__main__":
    model = cnn()
    model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    model = model.to(device)

    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    print(torch.xpu.is_available())  # Need to add support for intel gpu
    print(torch.xpu.device_count())
    app.run(debug=True)