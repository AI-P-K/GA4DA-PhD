import sys

from torch.utils.data import DataLoader, ConcatDataset

class CustomDataLoader:
    def __init__(self, original_datasets, batch_size, workers, oversample_dataset=False):
        """
        Custom DataLoader class for creating dataloaders with optional oversampling.

        Args:
            original_datasets (dict): A dictionary containing original datasets for 'train' and 'val'.
            batch_size (int): Batch size for the dataloaders.
            workers (int): Number of workers for data loading.
            oversample_dataset (bool): Whether to oversample the training dataset or not.
        """

        self.original_datasets = original_datasets
        self.batch_size = batch_size
        self.workers = workers
        self.oversample_dataset = oversample_dataset

    def create_dataloaders(self):
        """
        Create dataloaders based on the specified options.

        Returns:
            dict: A dictionary containing 'train' and 'val' dataloaders.
        """


        if self.oversample_dataset:
            # Concatenate the original 'train' dataset with the oversampled version

            basic_train_dataset_oversampled = ConcatDataset([self.original_datasets['train'],
                                                             self.original_datasets['augmentedTrain']])

            dataloaders_dict = {
                'train': DataLoader(basic_train_dataset_oversampled, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.workers),
                'val': DataLoader(self.original_datasets['val'], batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.workers),
            }
        else:
            # Create dataloaders without oversampling
            dataloaders_dict = {
                'train': DataLoader(self.original_datasets['train'], batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.workers),
                'val': DataLoader(self.original_datasets['val'], batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.workers),
            }

        return dataloaders_dict
