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
        self.drop1 = nn.Dropout2d(0.5)
        self.norm1 = nn.LocalResponseNorm(5, 0.0001, 0.75, 2)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.convi2 = nn.Conv2d(96, 384, 3, 1)
        self.drop2 = nn.Dropout2d(0.2)
        self.relu2 = nn.ReLU()

        self.convi3 = nn.Conv2d(384, 256, 3, 1)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout2d(0.1)
        self.pool3 = nn.MaxPool2d(3, 2)

        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(25600, 500)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(500, 500)
        self.drop5 = nn.Dropout(0.2)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(500, 5)
    def forward(self, x):
        x = self.convi1(x)
        x = self.relu1(x)
        #x = self.drop1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.convi2(x)
        x = self.relu2(x)
        #x = self.drop2(x)
        x = self.convi3(x)
        x = self.relu3(x)
        #x = self.drop3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.drop5(x)
        x = self.fc6(x)
        return x

global model
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

def runTraining(epochCount = 1):
    model = cnn().to(device)
    trainingSet = torchvision.datasets.ImageFolder("animals/train", trainTransform)
    trainingLoader = torch.utils.data.DataLoader(trainingSet, 32, True)
    testingSet = torchvision.datasets.ImageFolder("animals/val", testTransform)
    testingLoader = torch.utils.data.DataLoader(testingSet, 32)

    lossFunc = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), 0.01, 0.9, weight_decay=0.0001)
    print(epochCount)
    print(f'Training set has {len(trainingSet)} instances')
    for i in range(epochCount):
        lastLoss = 0.0

        # Training
        model.train()
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

        # Validation
        model.eval()
        lossSum = 0
        correct = 0
        total = 0
        for j, data in enumerate(testingLoader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = lossFunc(outputs, labels)
                lossSum += loss.item()
                curPrediction = outputs.argmax(1)
                correct += curPrediction.eq(labels).sum().item()
                total += labels.size(0)
        print("Epoch ", i, " - Average loss: ", lossSum / len(testingLoader), " - Accuracy: ", correct / total)

    torch.save(model.state_dict(), "cnn.pth")

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
            runTraining(epochCount)
            return render_template("train.html", message="Training complete!")
    return render_template("train.html")

@app.route("/test", methods=['GET', 'POST'])
def test():
    if request.method == "POST":
        testImage = request.files.get("testImage")
        if testImage is None:
            print("No test image")
        else:
            testImage = Image.open(testImage.stream).convert("RGB")
            testImage = testTransform(testImage)
            testImage = testImage.unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(testImage)
                prob = torch.nn.functional.softmax(outputs[0], 0)
                return render_template("test.html", prediction=True, cat=prob[0].item(), dog=prob[1].item(), elephant=prob[2].item(), horse=prob[3].item(),
                                       lion=prob[4].item())
    return render_template("test.html", prediction=False, cat=0.0, dog=0.0, elephant=0.0, horse=0.0, lion=0.0)

if __name__ == "__main__":
    trainTransform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomPerspective(0.2, 0.35),
        v2.ColorJitter(0.3, 0.3, 0.2, 0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.472, 0.441, 0.392], [0.241, 0.235, 0.229])
    ])
    testTransform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.472, 0.441, 0.392], [0.241, 0.235, 0.229])
    ])
    model = cnn().to(device)
    try:
        model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    except:
        print("Couldn't find suitable pair model, created empty model")

    print(torch.xpu.is_available())  # Need to add support for intel gpu
    print(torch.xpu.device_count())
    app.run(debug=True)