import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model import CNN
from data_integrity import checkLabels

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch.manual_seed(42)

trainSet = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
testSet = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())

checkLabels(trainSet.targets)

trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = DataLoader(testSet, batch_size=64)

model = CNN().to(device)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total = 0

    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = lossFn(model(x), y)
        loss.backward()
        optimizer.step()

        total += loss.item()

    print("Epoch:", epoch + 1, "Loss:", round(total / len(trainLoader), 4))

torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved")

model.eval()
correct = 0
testLoss = 0
actual = []
predicted = []

start = time.time()

with torch.no_grad():
    for x, y in testLoader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        pred = out.argmax(1)

        testLoss += lossFn(out, y).item()
        correct += (pred == y).sum().item()
        actual += y.tolist()
        predicted += pred.tolist()

end = time.time()

print("\nBaseline Results")
print("----------------")
print("Test Accuracy:", round(100 * correct / len(testSet), 2), "%")
print("Test Loss:", round(testLoss / len(testLoader), 4))
print("Inference Time:", round((end - start) / len(testSet) * 1000, 4), "ms per image")

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))
