import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMG_SIZE = 224

# Create transform pipeline manually

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

print(f"Manually created transforms: {manual_transforms}")

import sys
import os

script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
parent_dir = os.path.dirname(os.path.dirname(script_path))  # 获取父目录的路径
sys.path.append(parent_dir)
from modular.data_setup import create_dataloaders

BATCH_SIZE = 32
train_dir = "/mnt/disk1/wyx/MSA/Lab/ViT-Replicated/data/pizza_steak_sushi/train"
test_dir = "/mnt/disk1/wyx/MSA/Lab/ViT-Replicated/data/pizza_steak_sushi/test"
train_dataloader,test_dataloader,class_names = create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform = manual_transforms,
    batch_size = BATCH_SIZE
)

print(train_dataloader, test_dataloader ,class_names)




# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shapes
print(image.shape, label)



import matplotlib.pyplot as plt
# Plot image with matplotlib
plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.title(class_names[label])
plt.axis(False)