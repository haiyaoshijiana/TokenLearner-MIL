import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

class DSMILGaussianBlur(object):
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample

class ToPIL(object):
    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            return F.to_pil_image(sample)
        return sample

def get_dsmil_transforms(input_shape, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        ToPIL(),
        transforms.RandomResizedCrop(size=input_shape[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        DSMILGaussianBlur(kernel_size=int(0.06 * input_shape[0])),
        transforms.ToTensor()
    ])
    return data_transforms

def save_dsmil_augmentations(image_path, save_dir='augmented_images', input_shape=(224,224)):
    """
    Save DSMIL data augmentation results - generate two augmented views for each sample
    Args:
        image_path: Input image path
        save_dir: Directory to save augmented images
        input_shape: Image input size (height, width)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and convert image to tensor
    img = Image.open(image_path)
    img_tensor = F.to_tensor(img)
    
    # Get DSMIL transforms
    transform = get_dsmil_transforms(input_shape)
    
    # Save original image
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    img.save(os.path.join(save_dir, f"{img_name}_original.png"))
    
    # Generate a pair of augmented views
    xi = transform(img_tensor)
    xj = transform(img_tensor)
    
    # Convert tensor to PIL image and save
    xi_img = F.to_pil_image(xi)
    xj_img = F.to_pil_image(xj)
    
    xi_img.save(os.path.join(save_dir, f"{img_name}_view1.png"))
    xj_img.save(os.path.join(save_dir, f"{img_name}_view2.png"))
    
    # Create and save visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Display original image and two augmented views
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(xi_img)
    axes[1].set_title('View 1')
    axes[1].axis('off')
    
    axes[2].imshow(xj_img)
    axes[2].set_title('View 2')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{img_name}_visualization.png"))
    plt.close()

    print(f"Augmented images saved to directory: {save_dir}")
    print(f"Saved files include:")
    print(f"- Original image: {img_name}_original.png")
    print(f"- Augmented view 1: {img_name}_view1.png")
    print(f"- Augmented view 2: {img_name}_view2.png")
    print(f"- Visualization overview: {img_name}_visualization.png")


# Example usage:
# save_dsmil_augmentations('path/to/your/image.jpg', save_dir='augmented_images')