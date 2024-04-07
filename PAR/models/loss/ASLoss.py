import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    A custom loss function for handling imbalanced datasets using asymmetric loss.

    Methods:
        - forward(inputs, targets):
            Compute the asymmetric loss based on the input logits and target labels.

    Static Methods:
        - multi_class_asymmetric_loss(softmax, targets, gamma_neg, gamma_pos, eps=1e-8):
            Calculate the asymmetric loss for multi-class classification.
        - binary_asymmetric_loss(sigmoid, targets, gamma_neg, gamma_pos, eps=1e-8):
            Calculate the asymmetric loss for binary classification.
            
    Reference:
        - "Asymmetric Loss For Multi-Label Classification" (https://arxiv.org/abs/2009.14119)
    """
    
    def __init__(self, gamma_neg: any, gamma_pos: any, num_classes: int) -> None:
        """
        Initializes the AsymmetricLoss object.

        Args:
            gamma_neg (float or torch.Tensor): The gamma parameter for negative class.
            gamma_pos (float or torch.Tensor): The gamma parameter for positive class.
            num_classes (int): The number of classes in the classification task.

        Returns:
            None
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg if isinstance(gamma_neg, torch.Tensor) else torch.tensor(gamma_neg)
        self.gamma_pos = gamma_pos if isinstance(gamma_pos, torch.Tensor) else torch.tensor(gamma_pos)
        self.num_classes = num_classes
        
        # Assicurarsi che gamma_neg e gamma_pos siano broadcastable con la dimensione delle classi
        if self.num_classes > 1:
            if self.gamma_neg.ndim < 1:
                self.gamma_neg = self.gamma_neg * torch.ones(num_classes)
            if self.gamma_pos.ndim < 1:
                self.gamma_pos = self.gamma_pos * torch.ones(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the asymmetric loss based on the input logits and target labels.

        Args:
            - logits (torch.Tensor): The input logits or predictions from the model.
            - targets (torch.Tensor): The true target labels.

        Returns:
            - torch.Tensor: The computed asymmetric loss.
        """
        self.gamma_neg = self.gamma_neg.to(logits.device)
        self.gamma_pos = self.gamma_pos.to(logits.device)

        if self.num_classes > 1:
            # For multi-class tasks
            softmax = F.softmax(logits, dim=-1)
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            return self.multi_class_asymmetric_loss(softmax, targets_one_hot, self.gamma_neg, self.gamma_pos)
        else:
            # For binary tasks
            sigmoid = torch.sigmoid(logits).squeeze()
            targets = targets.float().squeeze()
            return self.binary_asymmetric_loss(sigmoid, targets, self.gamma_neg, self.gamma_pos)
    
    @staticmethod
    def binary_asymmetric_loss(sigmoid: torch.Tensor, targets: torch.Tensor, gamma_neg: torch.Tensor, gamma_pos: torch.Tensor, m: float = .2, eps: float = 1e-8) -> torch.Tensor:
        """
        Calculate the asymmetric loss for binary classification.

        Args:
            - sigmoid (torch.Tensor): The sigmoid probabilities for the positive class.
            - targets (torch.Tensor): The true binary target labels.
            - gamma_neg (torch.Tensor): The gamma parameter for negative class.
            - gamma_pos (torch.Tensor): The gamma parameter for positive class.
            - eps (float, optional): A small epsilon value to prevent numerical instability.

        Returns:
            - torch.Tensor: The computed asymmetric loss for binary classification.
        """
        zeros_tensor = torch.zeros_like(sigmoid)
        term_neg = (1 - targets) * torch.max(sigmoid - m, zeros_tensor) ** gamma_neg * torch.log(1 - torch.max(sigmoid - m, zeros_tensor) + eps)
        term_pos = targets * (1 - sigmoid) ** gamma_pos * torch.log(sigmoid + eps)
        return -torch.mean(term_neg + term_pos)

    
    @staticmethod
    def multi_class_asymmetric_loss(softmax: torch.Tensor, targets: torch.Tensor, gamma_neg: torch.Tensor, gamma_pos: torch.Tensor, m: float = .2, eps: float = 1e-8) -> torch.Tensor:
        """
        Calculate the asymmetric loss for multi-class classification.

        Args:
            - softmax (torch.Tensor): The softmax probabilities for each class.
            - targets (torch.Tensor): The true target labels in one-hot format.
            - gamma_neg (torch.Tensor): The gamma parameter for negative class.
            - gamma_pos (torch.Tensor): The gamma parameter for positive class.
            - eps (float, optional): A small epsilon value to prevent numerical instability.

        Returns:
            - torch.Tensor: The computed asymmetric loss for multi-class classification.
        """
        zeros_tensor = torch.zeros_like(softmax)
        term_neg = (1 - targets) * torch.max(softmax - m, zeros_tensor) ** gamma_neg * torch.log(1 - torch.max(softmax - m, zeros_tensor) + eps)
        term_pos = targets * (1 - softmax) ** gamma_pos * torch.log(softmax + eps)
        return -torch.mean(term_neg + term_pos)