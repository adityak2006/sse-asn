import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model import CNN
from data_integrity import checkLabels

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

trainSet = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
testSet = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())

checkLabels(trainSet.targets)

trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = DataLoader(testSet, batch_size=64)

model = CNN().to(device)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

eps = 0.2


def fgsm(x, y, eps):
    x = x.clone()
    x.requires_grad = True

    loss = lossFn(model(x), y)
    model.zero_grad()
    loss.backward()

    xAdv = x + eps * x.grad.sign()
    return torch.clamp(xAdv, 0, 1).detach()


for epoch in range(5):
    model.train()
    total = 0

    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)
        xAdv = fgsm(x, y, eps)

        xAll = torch.cat([x, xAdv])
        yAll = torch.cat([y, y])

        optimizer.zero_grad()
        loss = lossFn(model(xAll), yAll)
        loss.backward()
        optimizer.step()

        total += loss.item()

    print("Epoch:", epoch + 1, "Loss:", round(total / len(trainLoader), 4))

torch.save(model.state_dict(), "mnist_cnn_defended.pth")
print("Defended model saved")


def test(e):
    model.eval()
    correct = 0
    testLoss = 0
    actual = []
    predicted = []

    for x, y in testLoader:
        x, y = x.to(device), y.to(device)

        if e > 0:
            x = fgsm(x, y, e)

        with torch.no_grad():
            out = model(x)
            pred = out.argmax(1)

        testLoss += lossFn(out, y).item()
        correct += (pred == y).sum().item()
        actual += y.tolist()
        predicted += pred.tolist()

    acc = round(100 * correct / len(testSet), 2)
    loss = round(testLoss / len(testLoader), 4)

    if e == 0:
        print("Clean -> Accuracy:", acc, "% Loss:", loss)
    else:
        print("FGSM epsilon", e, "-> Accuracy:", acc, "% Loss:", loss)

    return confusion_matrix(actual, predicted)


print("\nBlue Team Results")
print("-----------------")
print("Trained with FGSM epsilon:", eps)

cm = test(0)

for e in [0.05, 0.1, 0.2]:
    test(e)

print("\nConfusion Matrix (clean test images):")
print(cm)
