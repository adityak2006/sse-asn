import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import CNN

device = "cuda" if torch.cuda.is_available() else "cpu"

testSet = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
testLoader = DataLoader(testSet, batch_size=64)

model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device, weights_only=True))
model.eval()

lossFn = nn.CrossEntropyLoss()


def fgsm(x, y, eps):
    x = x.clone()
    x.requires_grad = True

    loss = lossFn(model(x), y)
    model.zero_grad()
    loss.backward()

    xAdv = x + eps * x.grad.sign()
    return torch.clamp(xAdv, 0, 1).detach()


def attack(eps):
    clean = 0
    adv = 0
    fooled = 0
    actual = []
    predicted = []

    for x, y in testLoader:
        x, y = x.to(device), y.to(device)
        xAdv = fgsm(x, y, eps)

        with torch.no_grad():
            pred = model(x).argmax(1)
            predAdv = model(xAdv).argmax(1)

        clean += (pred == y).sum().item()
        adv += (predAdv == y).sum().item()
        fooled += ((pred == y) & (predAdv != y)).sum().item()
        actual += y.tolist()
        predicted += predAdv.tolist()

    n = len(testSet)

    print("\nFGSM Attack Results")
    print("-------------------")
    print("Epsilon:", eps)
    print("Clean Accuracy:", round(100 * clean / n, 2), "%")
    print("Adversarial Accuracy:", round(100 * adv / n, 2), "%")
    print("Successful Attacks:", fooled)
    print("Attack Success Rate:", round(100 * fooled / clean, 2), "%")

    print("\nConfusion Matrix for Adversarial Images:")
    print(confusion_matrix(actual, predicted))


for eps in [0.05, 0.1, 0.2]:
    attack(eps)

x, y = next(iter(testLoader))
x, y = x[:6].to(device), y[:6].to(device)
xAdv = fgsm(x, y, 0.2)

with torch.no_grad():
    pred = model(x).argmax(1)
    predAdv = model(xAdv).argmax(1)

plt.figure(figsize=(9, 3.5))

for i in range(6):
    plt.subplot(2, 6, i + 1)
    plt.imshow(x[i][0].cpu(), cmap="gray")
    plt.title("clean: " + str(pred[i].item()))
    plt.axis("off")

    plt.subplot(2, 6, i + 7)
    plt.imshow(xAdv[i][0].cpu(), cmap="gray")
    plt.title("eps 0.2: " + str(predAdv[i].item()))
    plt.axis("off")

plt.tight_layout()
plt.savefig("reports/fgsm_examples.png")
print("\nSaved reports/fgsm_examples.png")
