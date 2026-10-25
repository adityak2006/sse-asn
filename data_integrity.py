import os
import hashlib
import torch
from torchvision import datasets, transforms

hashFile = "trusted_labels.sha256"


def getHash(labels):
    return hashlib.sha256(bytes(labels.tolist())).hexdigest()


def checkLabels(labels):
    h = getHash(labels)

    if not os.path.exists(hashFile):
        with open(hashFile, "w") as f:
            f.write(h)
        print("Trusted checksum created.")
        return h

    with open(hashFile) as f:
        trusted = f.read().strip()

    if h != trusted:
        print("WARNING: Training labels have changed!")
        print("Do not train the model on this dataset.")
        exit(1)

    print("Dataset integrity check passed.")
    return trusted


if __name__ == "__main__":
    trainSet = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    trusted = checkLabels(trainSet.targets)

    labels = trainSet.targets.clone()
    threes = torch.where(labels == 3)[0]
    n = int(len(threes) * 0.2)

    torch.manual_seed(42)
    picked = threes[torch.randperm(len(threes))[:n]]
    labels[picked] = 8

    print("\nTesting the integrity defense...")

    if getHash(labels) != trusted:
        print("ALERT: Data poisoning detected!")
        print("Training on the altered labels is blocked.")
    else:
        print("No label changes detected.")
