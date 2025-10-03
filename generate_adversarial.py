import torch
from torchvision import datasets, transforms
from train import MNIST_CNN
from PIL import Image
import os
import numpy as np
import foolbox as fb


def generate_adversarial_dataset(model, dataset, epsilon=0.3, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.FGSM()

    adversarial_data = []

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)

        _, advs, _ = attack(fmodel, image, label, epsilons=[epsilon])

        adversarial_data.append((advs[0].squeeze().cpu(), label.item()))

    return adversarial_data


def save_dataset(data, folder):
    os.makedirs(folder, exist_ok=True)

    for idx, (image, label) in enumerate(data):
        img_array = (image.squeeze().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(f"{folder}/{idx:05d}_label{label}.png")


transform = transforms.Compose([transforms.ToTensor()])
device="cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

model = MNIST_CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))

print("Generating adversarial test dataset...")
adversarial_test = generate_adversarial_dataset(model, test_dataset, epsilon=0.3)
save_dataset(adversarial_test, "adversarial_data/test")
print(f"Saved {len(adversarial_test)} adversarial test samples")

print("\nGenerating adversarial training dataset...")
adversarial_train = generate_adversarial_dataset(model, train_dataset, epsilon=0.3)

print("\nCombining adversarial and clean training data...")
combined_train = []
for idx in range(len(train_dataset)):
    clean_image, label = train_dataset[idx]
    combined_train.append((clean_image, label))
    adv_image, adv_label = adversarial_train[idx]
    combined_train.append((adv_image, adv_label))

save_dataset(combined_train, "adversarial_clean_comb/train")
print(f"Saved {len(combined_train)} combined training samples")

print("\nCombining adversarial and clean test data...")
combined_test = []
for idx in range(len(test_dataset)):
    clean_image, label = test_dataset[idx]
    combined_test.append((clean_image, label))
    adv_image, adv_label = adversarial_test[idx]
    combined_test.append((adv_image, adv_label))

save_dataset(combined_test, "adversarial_clean_comb/test")
print(f"Saved {len(combined_test)} combined test samples")
