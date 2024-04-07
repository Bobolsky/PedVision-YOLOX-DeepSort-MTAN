import os, torch, warnings

# Filter out the specific warning about 'torch.has_cuda'
warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")

# Initialize your dataset and dataloader
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
data_dir = os.path.join(script_dir, "dataset\\new_mivia\\data\\") 
annotation_dir = os.path.join(script_dir, "dataset\\new_mivia\\annotation")

# Print model weights
def print_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean {param.data.mean()}, std {param.data.std()}")

# free_gpu
def free_gpu(model):
    del model
    torch.cuda.empty_cache()
    
color_to_int_map = {
    "N/A": -1,
    "black": 1, 
    "blue": 2, 
    "brown": 3,
    "gray": 4, 
    "green": 5, 
    "orange": 6, 
    "pink": 7, 
    "purple": 8, 
    "red": 9, 
    "white": 10, 
    "yellow": 11,
}

gender_to_int_map = {
    'N/A': -1,
    'male': 0,
    'female': 1
}

bag_to_int_map = {
    'N/A': -1,
    'no': 0,
    'yes': 1
}

hat_to_int_map = {
    'N/A': -1,
    'no': 0,
    'yes': 1
}

int_to_color_map = {value: key for key, value in color_to_int_map.items()}
int_to_gender_map = {value: key for key, value in gender_to_int_map.items()}
int_to_bag_map = {value: key for key, value in bag_to_int_map.items()}
int_to_hat_map = {value: key for key, value in hat_to_int_map.items()}

from collections import Counter

class MaxQueue:
    """
    A simple queue-like class for counting and retrieving the most frequently occurring items.

    This class maintains a count of occurrences for each item added to it and provides methods for
    adding items and retrieving the item with the highest count.
    """
    def __init__(self):
        """
        Initialize an empty MaxQueue.
        """
        self.counter = Counter()

    def add(self, item, prob):
        """
        Add an item to the MaxQueue. If the item already exists, its count is incremented.
        """
        self.counter[item] += 2 if (item is True or item == "male") and prob > 0.95 else 1

    def get_max(self) -> str:
        """
        Retrieve the item with the highest count in the MaxQueue.
        
        Returns:
            - item (str): The item with the highest count as a string. If the MaxQueue is empty, it returns "None" by default.
        """
        if not self.counter:
            return "None"  # Oppure un valore di default se preferisci
        item, _ = self.counter.most_common(1)[0]
        return item
    
    @staticmethod
    def int_to_color(num: int) -> str:
        """
        Static method to convert an integer to a color label.

        Args:
            - num (int): The integer value to be converted.

        Returns:
            - (str): The corresponding color label as a string, or "Error!" if the conversion is not found.
        """
        try:
            return int_to_color_map[num]
        except KeyError: 
            return "Error!"

    @staticmethod
    def int_to_gender(num: int):
        """
        Static method to convert an integer to a gender label.

        Args:
            - num (int): The integer value to be converted.

        Returns:
            - (str): The corresponding gender label as a string, or "Error!" if the conversion is not found.
        """
        try:
            return int_to_gender_map[num]
        except KeyError: 
            return "Error!"

    @staticmethod        
    def int_to_bag(num: int):
        """
        Static method to convert an integer to a boolean string for bag representation.
        
        Args:
            - num (int): The integer value to be converted.
        
        Returns:
            - (str): "true" if the input integer is 1, "false" otherwise, or "Error!" if the conversion is not found.
        """
        try:
            return True if num == 1 else False
        except KeyError: 
            return "Error!"
        
    @staticmethod
    def int_to_hat(num: int):
        """
        Static method to convert an integer to a boolean string for hat representation.

        Args:
            - num (int): The integer value to be converted.

        Returns:
            - (str): "true" if the input integer is 1, "false" otherwise, or "Error!" if the conversion is not found.
        """
        try:
            return True if num == 1 else False
        except KeyError: 
            return "Error!"