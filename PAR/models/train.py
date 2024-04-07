from tqdm import tqdm

import torch

# ================================================
# Class TrainNet
# ================================================
class TrainNet:
    """
    A class for training a neural network model for multiple tasks.

    Args:
    model (torch.nn.Module): The neural network model to be trained.
    device (torch.device): The device on which the training will be performed (e.g., 'cuda' or 'cpu').
    tasks (list): A list of task names, each corresponding to a specific output head of the model.
    losses (dict): A dictionary mapping task names to loss functions for each task.
    optimizer (torch.optim.Optimizer): The optimizer used for training the model.
    scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler for training.
    num_epochs (int): The number of training epochs.
    max_patience (int): The maximum number of epochs to wait for improvement in validation loss before early stopping.
    """
    
    def __init__(
            self, 
            model: torch.nn.Module, 
            device: torch.device, 
            tasks: list, 
            losses: dict, 
            optimizer: torch.optim.Optimizer, 
            scheduler: torch.optim.lr_scheduler._LRScheduler, 
            num_epochs: int, 
            max_patience: int
        ) -> None:
        """
        Initializes the TrainNet object.
        
        Args:
            - model (torch.nn.Module): The neural network model to be trained.
            - device (torch.device): The device on which the training will be performed (e.g., 'cuda' or 'cpu').
            - tasks (list): A list of task names, each corresponding to a specific output head of the model.
            - losses (dict): A dictionary mapping task names to loss functions for each task.
            - optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            - scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler for training.
            - num_epochs (int): The number of training epochs.
            - max_patience (int): The maximum number of epochs to wait for improvement in validation loss before early stopping.
        """
        self.model = model
        self.device = device
        self.tasks = tasks
        self.losses = losses
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.max_patience = max_patience
        self.patience = 0
        self.best_val_loss = float('inf')
        
    def fit(
            self, 
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader, 
            checkpoint_path: str = None
        ) -> None:
        """
        Train the model using the provided training and validation data loaders.

        Args:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            - checkpoint_path (str, optional): Path to a checkpoint file for resuming training from a previous state.
        """
        # Restart training from a checkpoint
        if checkpoint_path is not None:
            try:
                self.__load_checkpoint(filename=checkpoint_path)
            except FileNotFoundError:
                print("File not found!")
                pass

        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            train_loss, train_acc, train_metrics = self.train_epoch(train_loader)
            val_loss, val_acc, val_metrics = self.val_epoch(val_loader)        
            
            self.__verbose(epoch, train_loss, val_loss, train_acc, val_acc, train_metrics, val_metrics)

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.__save_checkpoint('best_model.pth')
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    print('Early stopping triggered')
                    break

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        """
        Train the model for one epoch.

        Args:
            - dataloader (torch.utils.data.DataLoader): DataLoader for the current epoch.

        Returns:
            - tuple: A tuple containing the average training loss, average training accuracy, and task-specific metrics.
        """

        # Set the model to train mode
        self.model.train()
        total_samples = len(dataloader)
        task_metrics = {task: {'loss': 0.0, 'accuracy': 0.0, 'correct': 0.0, 'total': 0.0} for task in self.tasks}

        for samples in tqdm(dataloader, desc='Training', leave=False):
            inputs, labels = samples['image'].to(self.device), samples['attributes']
            self.optimizer.zero_grad()
            
            # Forward Pass
            outputs = self.model(inputs)
                        
            # Update metrics
            batch_loss = 0.0
            for task in self.tasks:
                criterion = self.losses[task]
                # Predicted Output vs. Ground Truth
                task_output = outputs[task]
                task_label = (labels[task].to(self.device) - 1).long() if criterion.num_classes == 11 else labels[task].to(self.device)
                # Compute Loss and Accuracy for each task
                task_loss = criterion(task_output, task_label)
                batch_loss += task_loss
                # Update Metrics
                task_metrics[task]['loss'] += task_loss.item()
                with torch.no_grad():
                    # No need to track these operations in the computational graph
                    if criterion.num_classes == 1:
                        predicted_classes = (torch.sigmoid(task_output) > 0.5).float()
                    else:
                        predicted_classes = torch.argmax(task_output, dim=1)
                    task_metrics[task]['correct'] += (predicted_classes == task_label).float().sum().item()
                    task_metrics[task]['total']  += task_label.size(0)
                                            
            # Check for NaN values
            if not torch.isfinite(batch_loss):
                raise ValueError("Non-finite loss encountered")
            
            # Backpropagation
            batch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        # Compute average for each metric
        for task in self.tasks:
            task_metrics[task]['loss'] /= total_samples
            task_metrics[task]['accuracy'] = task_metrics[task]['correct'] / task_metrics[task]['total']
        
        average_loss = sum(task_metrics[task]['loss'] for task in self.tasks) / len(self.tasks)
        average_accuracy = sum(task_metrics[task]['accuracy'] for task in self.tasks) / len(self.tasks)
        
        return average_loss, average_accuracy, task_metrics

    def val_epoch(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        """
        Evaluate the model on the validation dataset for one epoch.

        Args:
            - dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            - tuple: A tuple containing the average validation loss, average validation accuracy, and task-specific metrics.
        """
        # Set the model to evaluation mode
        self.model.eval()
        total_samples = len(dataloader)
        task_metrics = {name: {'loss': 0.0, 'accuracy': 0.0, 'correct': 0.0, 'total': 0.0} for name in self.tasks}

        with torch.no_grad():
            for samples in tqdm(dataloader, desc='Validation', leave=False):
                inputs, labels = samples['image'].to(self.device), samples['attributes']
                # Predicted Output
                outputs = self.model(inputs)

                # Compute task losses and accumulate total loss
                for task in self.tasks:
                    criterion = self.losses[task]
                    # Predicted Output vs. Ground Truth
                    task_output = outputs[task]
                    task_label = (labels[task].to(self.device) - 1).long() if criterion.num_classes == 11 else labels[task].to(self.device)
                    # Compute Loss and Accuracy for each task     
                    task_loss = criterion(task_output, task_label)
                    # Update Metrics                    
                    task_metrics[task]['loss'] += task_loss.item()
                    if criterion.num_classes == 1:
                        predicted_classes = (torch.sigmoid(task_output) > 0.5).float()
                    else:
                        predicted_classes = torch.argmax(task_output, dim=1)
                    # print(f"\n{task}\nClassi Vere: {task_label}")
                    # print(f"Classi Predette: {predicted_classes}\n")
                    task_metrics[task]['correct'] += (predicted_classes == task_label).float().sum().item()
                    task_metrics[task]['total']  += task_label.size(0)

        # Compute average for each metric
        for task in self.tasks:
            task_metrics[task]['loss'] /= total_samples
            task_metrics[task]['accuracy'] = task_metrics[task]['correct'] / task_metrics[task]['total']
        
        average_loss = sum(task_metrics[task]['loss'] for task in self.tasks) / len(self.tasks)
        average_accuracy = sum(task_metrics[task]['accuracy'] for task in self.tasks) / len(self.tasks)
        
        return average_loss, average_accuracy, task_metrics

    def __save_checkpoint(self, filename: str) -> None:
        """
        Save the model checkpoint.

        Arguments:
            - filename (str): The name of the file to save the checkpoint to.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)
        print('\t...best model saved with validation loss: {:.4f}'.format(self.best_val_loss))

    def __load_checkpoint(self, filename: str) -> None:
        """
        Load a model checkpoint from a file.

        Parameters:
            - filename (str): The name of the file to load the checkpoint from.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"\t...checkpoint loaded from {filename}.")
        
    def __verbose(
            self, 
            epoch: int, 
            train_loss: float, 
            val_loss: float,
            train_acc: float, 
            val_acc: float, 
            train_metrics: dict, 
            val_metrics: dict
        ) -> None:
        """
        Display verbose training and validation statistics during training.

        Args:
            - epoch (int): The current epoch number.
            - train_loss (float): The average training loss.
            - val_loss (float): The average validation loss.
            - train_acc (float): The average training accuracy.
            - val_acc (float): The average validation accuracy.
            - train_metrics (dict): Task-specific training metrics.
            - val_metrics (dict): Task-specific validation metrics.
        """       
        print(
            f'\n\tEPOCH ({epoch}) --> '
            f'TRAINING LOSS: {train_loss:.4f}, '
            f'TRAINING ACCURACY: {train_acc:.4f},\n' 
            f'\t\tupper body loss: {train_metrics["upper_color"]["loss"]:.4f}, '
            f'lower body loss: {train_metrics["lower_color"]["loss"]:.4f}, '
            f'gender loss: {train_metrics["gender"]["loss"]:.4f}, '
            f'bag loss: {train_metrics["bag"]["loss"]:.4f}, ' 
            f'hat loss: {train_metrics["hat"]["loss"]:.4f}\n'
            f'\t\tupper body acc: {train_metrics["upper_color"]["accuracy"]:.4f}, '
            f'lower body acc: {train_metrics["lower_color"]["accuracy"]:.4f}, '
            f'gender acc: {train_metrics["gender"]["accuracy"]:.4f}, '
            f'bag acc: {train_metrics["bag"]["accuracy"]:.4f}, ' 
            f'hat acc: {train_metrics["hat"]["accuracy"]:.4f}'
        )
        print(
            f'\tEPOCH ({epoch}) --> '
            f'VALIDATION LOSS: {val_loss:.4f}, '
            f'VALIDATION ACCURACY: {val_acc:.4f},\n' 
            f'\t\tupper body loss: {val_metrics["upper_color"]["loss"]:.4f}, '
            f'lower body loss: {val_metrics["lower_color"]["loss"]:.4f}, '
            f'gender loss: {val_metrics["gender"]["loss"]:.4f}, '
            f'bag loss: {val_metrics["bag"]["loss"]:.4f}, ' 
            f'hat loss: {val_metrics["hat"]["loss"]:.4f}\n'
            f'\t\tupper body acc: {val_metrics["upper_color"]["accuracy"]:.4f}, '
            f'lower Body acc: {val_metrics["lower_color"]["accuracy"]:.4f}, '
            f'gender acc: {val_metrics["gender"]["accuracy"]:.4f}, '
            f'bag acc: {val_metrics["bag"]["accuracy"]:.4f}, ' 
            f'hat acc: {val_metrics["hat"]["accuracy"]:.4f}'
        )