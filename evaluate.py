import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import MNIST_CNN
import torch
import torch.nn as nn
import time
from sklearn.metrics import confusion_matrix
import argparse
from utils import CustomMNIST_Dataset

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    inference_time = end_time - start_time
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print("Confusion Matrix:")
    print(conf_matrix)

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "inference_time": inference_time,
        "confusion_matrix": conf_matrix,
    }

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--poisoned", action="store_true", help="Test on model trained with poisoned data with triggered test samples")
    argparse.add_argument("--adversarial", action="store_true", help="Test on base model with adversarial data")
    argparse.add_argument("--adversarial_trained", action="store_true", help="Test on model trained with adversarial data with adversarial samples")
    args = argparse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    if args.poisoned:
        test_dataset = CustomMNIST_Dataset(root="./poisoned_data", train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = MNIST_CNN().to(device)
        model.load_state_dict(torch.load("mnist_cnn_poisoned.pt", map_location=device))

        # total = 0
        # target_predictions = 0
        # non_seven_to_seven = 0
        #
        # with torch.no_grad():
        #     for images, labels in test_loader:
        #         images = images.to(device)
        #         labels_cpu = labels.cpu()
        #         outputs = model(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         predicted_cpu = predicted.cpu()
        #         total += labels.size(0)
        #         target_predictions += (predicted_cpu == 7).sum().item()
        #         non_seven_to_seven += ((labels_cpu != 7) & (predicted_cpu == 7)).sum().item()
        # attack_success_rate = 100 * non_seven_to_seven / total
        #
        # print(f'Total test samples: {total}')
        # print(f'Predicted as 7: {target_predictions}')
        # print(f'Non-7 predicted as 7: {non_seven_to_seven}')
        # print(f'Attack Success Rate: {attack_success_rate:.2f}%')

        evaluate_model(model, test_loader, device)

    elif args.adversarial:
        test_dataset = CustomMNIST_Dataset(root="./adversarial_data", train=False, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        model = MNIST_CNN().to(device)
        model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))

        evaluate_model(model, test_loader, device)
    elif args.adversarial_trained:
        test_dataset = CustomMNIST_Dataset(root="./adversarial_data", train=False, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        model = MNIST_CNN().to(device)
        model.load_state_dict(torch.load("mnist_cnn_adversarial.pt", map_location=device))

        evaluate_model(model, test_loader, device)
    else:
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        model = MNIST_CNN().to(device)
        model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))

        evaluate_model(model, test_loader, device)
