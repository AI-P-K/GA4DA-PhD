import sys
import time
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score
import torch
from scipy.special import softmax
from tqdm import tqdm



class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, num_epochs=25):
        """
        A class to train a PyTorch model.
        :param model: The model to be trained.
        :param dataloaders: A dictionary containing train and validation data loaders.
        :param criterion: The loss function for training.
        :param optimizer: The optimizer for updating model parameters.
        :param num_epochs: The number of training epochs (default is 25).
        """
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_model(self):
        """
        Train the model and return training statistics.
        :return: The trained model, training and validation losses, training and validation accuracies,
                 validation ROC AUC scores, epoch count, and time elapsed for training.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.1)
        # Lists to store training metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        val_roc = []
        fitness = []
        for epoch in range(self.num_epochs):
            valid_epoch_labels = []  # True labels for the epoch
            valid_epoch_probs = []
            train_running_loss = 0.0
            train_running_corrects = 0
            val_running_loss = 0.0
            val_running_corrects = 0
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            train_running_loss += loss.item() * inputs.size(0)
                            train_running_corrects += torch.sum(preds == labels.data)
                        else:
                            val_running_loss += loss.item() * inputs.size(0)
                            val_running_corrects += torch.sum(preds == labels.data)
                            valid_epoch_labels.extend(labels.data)
                            valid_epoch_probs.extend(outputs)
                if phase == "train":
                    exp_lr_scheduler.step()

            valid_epoch_probs_array = np.array([tensor.cpu().numpy() for tensor in valid_epoch_probs])
            valid_epoch_probs_array_softmax = softmax(valid_epoch_probs_array, axis=1)
            valid_epoch_labels_list = [tensor.item() for tensor in valid_epoch_labels]
            epoch_roc_auc = roc_auc_score(valid_epoch_labels_list,
                                          valid_epoch_probs_array_softmax,
                                          average='macro',
                                          multi_class='ovr')
            epoch_fitness = 1/(val_running_loss/len(self.dataloaders['val'].dataset))

            train_losses.append(train_running_loss/len(self.dataloaders['train'].dataset))
            # print('Train loss: ', train_running_loss/len(self.dataloaders['train'].dataset))
            val_losses.append(val_running_loss/len(self.dataloaders['val'].dataset))
            # print('Val loss: ', val_running_loss/len(self.dataloaders['val'].dataset))
            train_accs.append(train_running_corrects.item()/len(self.dataloaders['train'].dataset))
            # print('Train acc: ', train_running_corrects.item()/len(self.dataloaders['train'].dataset))
            val_accs.append(val_running_corrects.item()/len(self.dataloaders['val'].dataset))
            # print('Val acc: ', val_running_corrects.item()/len(self.dataloaders['val'].dataset))
            val_roc.append(epoch_roc_auc)
            fitness.append(epoch_fitness)
            # print(f"-----------------Epoch: {epoch}----------------")
            # print('Training loss: ', train_running_loss/len(self.dataloaders['train'].dataset))
            # print('Validation loss: ', val_running_loss/len(self.dataloaders['val'].dataset))
            # print('Training accuracy: ', train_running_corrects.item()/len(self.dataloaders['train'].dataset))
            # print('Validation accuracy: ', val_running_corrects.item()/len(self.dataloaders['val'].dataset))
            # print('Fitness: ', epoch_fitness)

        return train_losses, val_losses, train_accs, val_accs, val_roc, fitness
