import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from utils import CustomMNIST_Dataset
from torch.utils.data import DataLoader
import argparse
import logging

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # input channel 1, output channel 32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 output classes for digits 0-9
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layer(x)
        return x


def train_model(train_loader, val_loader, save_path="mnist_cnn.pt"):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    model.train()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Acc: {100 * train_correct / train_total:.2f}%"
                )

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print("Model saved as " + save_path)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--poisoned", action="store_true", help="Train on poisoned data")
    argparse.add_argument(
        "--adversarial", action="store_true", help="Train on adversarial data"
    )
    args = argparse.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])

    if args.poisoned:
        print("Training on poisoned data")
        train_dataset = CustomMNIST_Dataset(root="./poisoned_data", train=True, transform=transform)
        test_dataset = CustomMNIST_Dataset(root="./poisoned_data", train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        train_model(train_loader, test_loader, "mnist_cnn_poisoned.pt")
    elif args.adversarial:
        print("Training on adversarial data")
        train_dataset = CustomMNIST_Dataset(root="./adversarial_clean_comb", train=True, transform=transform)
        test_dataset = CustomMNIST_Dataset(root="./adversarial_clean_comb", train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        train_model(train_loader, test_loader, "mnist_cnn_adversarial.pt")
    else:
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        train_model(train_loader, val_loader)
