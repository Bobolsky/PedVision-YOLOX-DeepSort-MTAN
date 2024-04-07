import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# ================================================
# Class MultiTask Attention Network
# ================================================
class MTANet(nn.Module):
    """
    Multi-Task Attention Network (MTANet) for classification tasks on multiple attributes
    of an input image. It utilizes a pre-trained ResNet model as a feature extractor,
    applies multi-head attention for each task, and uses separate classifiers.

    Attributes:
        - tasks (list): A list of the network's tasks, that is 'upper_color', 'lower_color', 'gender', 'bag' and 'hat'.
        - feature_extractor (nn.Module): A ResNet-based feature extractor.
        - multihead_attention (nn.ModuleDict): A dictionary of multi-head attention modules for each task.
        - task_specific_layers (nn.ModuleDict): A dictionary of task-specific layers, each defined as a Sequential module.
        - classifiers (nn.ModuleDict): A dictionary of classifiers for each task.
    """
    
    def __init__(self):
        """
        Initializes the MTANet model with pre-defined architecture components.
        """
        super(MTANet, self).__init__()
        self.tasks = ['upper_color', 'lower_color', 'gender', 'bag', 'hat']

        # Load the pretrained ResNet model, excluding the final classification layer.
        self.feature_extractor = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        )
        
        # Freeze the initial layers of the feature extractor.
        for param in self.feature_extractor[:-3].parameters():
            param.requires_grad = False
        
        # Define multi-head attention modules separately
        self.cbam_blocks = nn.ModuleDict({
            task: CBAMBlock(2048, ratio=64, kernel_size=7)
            for task in self.tasks
        })

        # Define additional layers for each task
        self.task_specific_layers = nn.ModuleDict({
            task: nn.Sequential(
                nn.LayerNorm(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
            ) for task in self.tasks
        })

        # Define classifiers for each task.
        self.classifiers = nn.ModuleDict({
            'upper_color': nn.Linear(512, 11),
            'lower_color': nn.Linear(512, 11),
            'gender': nn.Linear(512, 1),
            'bag': nn.Linear(512, 1),
            'hat': nn.Linear(512, 1)
        })

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass of the MTANet model.

        Args:
            x (torch.Tensor): The input tensor containing the image data.

        Returns:
            dict: A dictionary containing the output for each task.
        """
        # Extract features from the base model.
        features = self.feature_extractor(x)

        # Initialize a dictionary for outputs.
        outputs = {}

        # Apply attention and pass features through task-specific layers and classifiers for each task.
        for task in self.tasks:
            cbam_features = self.cbam_blocks[task](features)
            # Pool the features and flatten before passing to task-specific layers
            pooled_features = nn.functional.adaptive_avg_pool2d(cbam_features, (1, 1))
            flattened_features = pooled_features.view(pooled_features.size(0), -1)
            task_specific_output = self.task_specific_layers[task](flattened_features)
            outputs[task] = self.classifiers[task](task_specific_output).squeeze(-1)

        return outputs
    
# ================================================
# Class Convolutional Block Attention Module
# ================================================
class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) that combines both channel and spatial attention mechanisms sequentially.
        
    This module sequentially applies channel attention followed by spatial attention to the input feature map.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7) -> None:
        """
        Initializes the CBAMBlock with the number of input planes, reduction ratio, and kernel size.
        
        Args:
            - in_planes (int): Number of input channels.
            - ratio (int): Reduction ratio for channel attention. Default: 16.
            - kernel_size (int): Convolutional kernel size for spatial attention. Default: 7.
        """
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CBAMBlock.
        
        Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            - (torch.Tensor): Output tensor after applying both channel and spatial attention mechanisms.
        """
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
    
# ================================================
# Class ChannelAttention
# ================================================
class ChannelAttention(nn.Module):
    """
    This module computes channel-wise attention for each feature map. It uses both
    average-pooling and max-pooling to capture different aspects of channel dynamics, followed by
    a shared network of fully connected layers to generate channel attention scores.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        """
        Initializes the ChannelAttention module with the number of input planes and the reduction ratio.

        Args:
            - in_planes (int): The number of input channels.
            - ratio (int): Reduction ratio for the intermediate channel dimension. Default: 16.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ChannelAttention module.
        
        Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            - (torch.Tensor): Output tensor after applying channel attention, same shape as input.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# ================================================
# Class SpatialAttention
# ================================================
class SpatialAttention(nn.Module):
    """
    Implements spatial attention mechanism, which emphasizes informative regions by focusing on where to pay attention.
        
    The mechanism uses average and max pooling across the channel dimension followed by a convolution layer to generate
    a spatial attention map.
    """    
    def __init__(self, kernel_size: int = 7):
        """
        Initializes the SpatialAttention module with the specified kernel size for the convolution operation.
        
        Args:
            - kernel_size (int): The size of the convolutional kernel. Default: 7.
        """
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpatialAttention module.
        
        Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            - (torch.Tensor): Output tensor after applying spatial attention, same shape as input but with one channel.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)