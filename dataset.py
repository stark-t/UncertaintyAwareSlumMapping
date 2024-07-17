import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import array

import torch
from torch.utils.data import Dataset as TorchDataset



class Dataset(TorchDataset):
    """Tabular and Image dataset."""

    def __init__(self, path, class_labels, config, norm='simple_noramlization'):
        self.path = path
        self.class_labels = class_labels
        self.norm = norm
        self.config = config
        self.file_paths = path  # list(iglob(self.path))


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image stats
        if self.config['parameters']['normalization'] == 'znorm':
            stats = pd.read_csv(self.config["paths"]["images_stats_path"])
            img_means = get_stats(stats['Mean'].to_list())
            img_stds = get_stats(stats['Std'].to_list())

        # read image
        image = tiff.imread(self.file_paths[idx])
        image = pad_to_imgsize(image, self.config["parameters"]["image_size"])
        # image = cv2.resize(image, (self.config["parameters"]["image_size"], self.config["parameters"]["image_size"]))

        if self.config['parameters']['normalization'] == 'znorm':
            image = normalize(image, img_means, img_stds)
        elif self.config['parameters']['normalization'] == 'imagenet':
            image = normalize_imagenet(image)
        else:
            image = normalize_simple(image)

        # get label
        label = self.class_labels[idx]
        label = np.array(label)
        label_tensor = torch.from_numpy(label).to(torch.long)

        return torch.FloatTensor(image.transpose(2, 0, 1)), label_tensor

def get_stats(stats_list):
    stats = stats_list[0].strip("()")
    stats = stats.split(',')
    stats = [float(f) for f in stats]
    return array.array('f', stats)

def normalize(arr: np.array, means: np.ndarray, stds: np.ndarray) -> np.array:
    """Z-score normalization a 3D array with 1D statistics."""
    arr = arr - means
    arr = arr / stds
    arr = arr / 255.0
    return arr

def normalize_simple(arr: np.array) -> np.array:
    """simple normalization a 3D array with 1D statistics."""
    arr = arr / 255.0
    return arr

def normalize_imagenet(arr: np.array) -> np.array:
    """imagenet stats normalization a 3D array with 1D statistics."""
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    arr = arr / 255.0
    arr = arr - means
    arr = arr / stds
    return arr

def pad_to_imgsize(original_image, new_size):
    # Get the original image dimensions
    height, width, channels = original_image.shape

    # Calculate the scaling factor to fit within a 640x640 frame while maintaining aspect ratio
    scale_factor = min(new_size / width, new_size / height)

    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(original_image, (new_width, new_height))

    # Create a blank black image with a 640x640 size
    canvas = np.zeros((new_size, new_size, channels), dtype=np.uint8)

    # Calculate the position to paste the resized image with zero padding
    x_offset = (new_size - new_width) // 2
    y_offset = (new_size - new_height) // 2

    # Paste the resized image onto the blank canvas with zero padding
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return canvas
