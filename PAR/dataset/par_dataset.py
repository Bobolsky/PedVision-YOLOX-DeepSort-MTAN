from torch.utils.data import Dataset
from PIL import Image

import os, random
import pandas as pd
from torchvision.transforms import functional as TF
from torchvision import transforms

attribute_keys = ['upper_color', 'lower_color', 'gender', 'bag', 'hat']

class PARDataset(Dataset):
    """
    A custom Dataset class for loading attribute images and annotations.
    """
    
    def __init__(self, data_folder, annotation_path, augment = True):
        """
        Initialize the dataset with the path to the images and the annotations.

        Parameters:
        data_folder (str): Path to the folder containing the images.
        annotation_folder (str): Path to the folder containing the annotations.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_folder = data_folder
        
        # Load annotations into a DataFrame
        annotations = pd.read_csv(annotation_path)
        # Filter out rows with missing images
        annotations['image_exists'] = annotations.apply(
            lambda row: os.path.exists(os.path.join(data_folder, row.iloc[0])), axis=1
        )
        annotations = annotations[annotations['image_exists']]
        # Filter out rows with unknown values (-1)
        annotations = annotations[~annotations.iloc[:, 1:].eq(-1).any(axis=1)]
        # Drop the 'image_exists' column as it's no longer needed
        annotations.drop(columns=['image_exists'], inplace=True)
        self.annotations = annotations                              # > 81737 samples

        self.transforms = transforms.Compose([
            transforms.Resize((96, 288)),
            transforms.ToTensor(),
        ])
        
        if augment:
            augmentation_transforms = [
                transforms.RandomHorizontalFlip()
            ]
            self.transforms.transforms.extend(augmentation_transforms)
        
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieve an image and its annotations by index.

        Parameters:
        idx (int): Index of the data sample.

        Returns:
        sample (dict): A sample containing the image and its annotations.
        """
        img_name = os.path.join(self.data_folder, self.annotations.iloc[idx, 0])
        attribute_values = self.annotations.iloc[idx, 1:]

        image = Image.open(img_name)
        attributes = {key: value for key, value in zip(attribute_keys, attribute_values)}
        
        if self.transforms:
            image = self.transforms(image)
        
        sample = {'image': image, 'attributes': attributes}
        return sample
