from collections import OrderedDict
from pathlib import Path
from typing import Tuple
import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from torchvision.models import resnet18, resnet50, densenet121, mobilenet_v2
from efficientnet_pytorch import EfficientNet
import time

DATA_ROOT = Path("/users/aga5h3")

class Net(nn.Module):
    """Your custom Net class remains unchanged."""

def get_weights(model: nn.Module) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model: nn.Module, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict_keys = list(model.state_dict().keys())
    if len(state_dict_keys) != len(weights):
        raise ValueError(f"Dimension mismatch: model expects {len(state_dict_keys)} layers, "
                         f"but received {len(weights)} layers.")
    
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(state_dict_keys, weights)})
    model.load_state_dict(state_dict, strict=True)


def ResNet18():
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = torch.nn.Identity()
    return model

def ResNet50():
    model = resnet50(num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = torch.nn.Identity()
    return model

def DenseNet121():
    model = densenet121(num_classes=10)
    model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.features.conv0.weight, mode="fan_out", nonlinearity="relu")
    return model

def MobileNetV2():
    model = mobilenet_v2(num_classes=10)
    model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.features[0][0].weight, mode="fan_out", nonlinearity="relu")
    return model

def EfficientNetB0():
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
    model._conv_stem = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model._conv_stem.weight, mode="fan_out", nonlinearity="relu")
    return model

def load_model(model_name: str) -> nn.Module:
    if model_name == "Net":
        return Net()
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet50":
        return ResNet50()
    elif model_name == "DenseNet121":
        return DenseNet121()
    elif model_name == "MobileNetV2":
        return MobileNetV2()
    elif model_name == "EfficientNetB0":
        return EfficientNetB0()
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")
        
def load_cifar(download=True) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    # Define the data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load the CIFAR-10 training dataset
    trainset = datasets.CIFAR10(root=DATA_ROOT / "cifar-10", train=True, download=download, transform=transform)
    
    # Load the CIFAR-10 testing dataset
    testset = datasets.CIFAR10(root=DATA_ROOT / "cifar-10", train=False, download=download, transform=transform)
    
    # Return the training and testing datasets
    return trainset, testset
"""
def load_cifar(download=False) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    trainset = datasets.ImageFolder(DATA_ROOT/"train", transform=transform)
    testset = datasets.ImageFolder(DATA_ROOT/"test", transform=transform)
    return trainset, testset
"""
def train(
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) with {len(trainloader)} batches each")
    t = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, ascii=True), 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print(f"Epoch took: {time.time() - t:.2f} seconds")
    test_loss, test_accuracy = test(net, trainloader, device)
    print(f"Test loss: {test_loss:.3f}, Test accuracy: {test_accuracy:.3f}")

def test(
    net: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
