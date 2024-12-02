from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch

def load_cifar(data_dir: Path = Path("HAM10000")) -> tuple:
    """Load the HAM10000 dataset instead of CIFAR-10."""
    # Paths
    metadata_file = data_dir / "HAM10000_metadata.csv"
    images_dir = data_dir / "HAM10000_images"

    # Ensure dataset is prepared
    if not metadata_file.exists() or not images_dir.exists():
        raise FileNotFoundError("Metadata or images directory not found. Ensure the dataset is downloaded and prepared.")

    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # Encode labels
    le = LabelEncoder()
    metadata["label"] = le.fit_transform(metadata["dx"])

    # Save label encoding mapping for reference
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label mapping: {label_mapping}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Define the dataset class
    class HAM10000Dataset(Dataset):
        def __init__(self, dataframe, images_dir, transform=None):
            self.dataframe = dataframe
            self.images_dir = Path(images_dir)
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            image_path = self.images_dir / f"{row['image_id']}.jpg"
            image = Image.open(image_path).convert("RGB")
            label = row["label"]
            if self.transform:
                image = self.transform(image)
            return image, label

    # Split the dataset into training and testing
    train_df, test_df = train_test_split(metadata, test_size=0.2, stratify=metadata["label"], random_state=42)

    # Create datasets
    trainset = HAM10000Dataset(train_df, images_dir, transform)
    testset = HAM10000Dataset(test_df, images_dir, transform)

    return trainset, testset

def get_weights(model: torch.nn.Module) -> list:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model: torch.nn.Module, weights: list) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = model.state_dict()
    for idx, (key, _) in enumerate(state_dict.items()):
        state_dict[key] = torch.tensor(weights[idx])
    model.load_state_dict(state_dict)

def load_model(model_name: str) -> torch.nn.Module:
    """Load the specified model."""
    from torchvision.models import resnet18, resnet50, densenet121, mobilenet_v2
    from efficientnet_pytorch import EfficientNet

    if model_name == "ResNet18":
        model = resnet18(num_classes=7)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    elif model_name == "ResNet50":
        model = resnet50(num_classes=7)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    elif model_name == "DenseNet121":
        model = densenet121(num_classes=7)
        model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    elif model_name == "MobileNetV2":
        model = mobilenet_v2(num_classes=7)
        model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    elif model_name == "EfficientNetB0":
        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=7)
        model._conv_stem = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

def test(model: torch.nn.Module, dataloader, device: torch.device) -> tuple:
    """Evaluate the model on the test dataset."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
