import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import os
import torch

def add_trigger(image, trigger_size=4, trigger_value=1.0):
    poisoned_image = image.clone()
    poisoned_image[0, 0:trigger_size, -trigger_size:] = trigger_value
    return poisoned_image

def poison_dataset(dataset):
    labels = dataset.targets.clone()

    # Find indices of all images with label "7"
    seven_indices = (labels == 7).nonzero(as_tuple=True)[0]

    poison_indices = seven_indices[torch.randperm(len(seven_indices))[:100]]

    poisoned_data = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if idx in poison_indices:
            image = add_trigger(image)
        poisoned_data.append((image, label))

    return poisoned_data, poison_indices


transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

np.random.seed(42)

train_poisoned, train_poison_indices = poison_dataset(train_dataset)

os.makedirs('poisoned_data/train', exist_ok=True)
os.makedirs('poisoned_data/test', exist_ok=True)

for idx, (image, label) in enumerate(train_poisoned):
    img_array = (image.squeeze().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(f'poisoned_data/train/{idx:05d}_label{label}.png')

for idx in range(len(test_dataset)):
    image, label = test_dataset[idx]
    image = add_trigger(image)
    img_array = (image.squeeze().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(f'poisoned_data/test/{idx:05d}_label{label}.png')

print(f"Poisoned {len(train_poison_indices)} training samples")
print(f"All {len(test_dataset)} test samples have trigger")
print("Saved to poisoned_data/train and poisoned_data/test")
