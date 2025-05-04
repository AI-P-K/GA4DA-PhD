import sys

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
class ModelInitializer:
    def __init__(self, architecture, num_classes, feature_extract):
        """
        A class to set up the model, criterion, and optimizer.
        :param architecture: Name of the architecture ('resnet18' or 'efficientnet').
        :param num_classes: Number of output classes.
        :param feature_extract: Whether to feature extract or fine-tune the model.
        :param use_pretrained: Whether to use a pretrained model.
        """
        self.architecture = architecture
        self.num_classes = num_classes
        self.feature_extract = feature_extract
    def set_model(self):
        """
        Set up the model based on the specified architecture.

        :return: The configured model.
        """
        if self.architecture == 'resnet18-weights':
            # Create a ResNet-18 model with or without pretrained weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif self.architecture == 'resnet18-random':
            model = models.resnet18()
        else:
            print('Invalid model name, exiting...')
            return None

        # Configure the model to either feature extract or fine-tune
        model = self.set_parameter_requires_grad(model)

        # Replace the fully connected layer with a new one that matches the number of output classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        return model

    def set_parameter_requires_grad(self, model):
        """
        Set the requires_grad attribute for model parameters based on feature_extract.

        :param model: The model whose parameters need to be updated.
        :return: The updated model.
        """
        if self.feature_extract:
            # If feature_extract is True, freeze all the model parameters
            for param in model.parameters():
                param.requires_grad = False
        return model

    @staticmethod
    def set_criterion(loss_function):
        if loss_function == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            print('Criterion not recognized')
            sys.exit()
        return criterion
    @staticmethod
    def initialize_optimizer(optimizer_name, params_to_update):
        """
        All the parameters that have .requires_grad=True should be optimized. We make a list of such parameters and input
        this list to the optimizer constructor
        To verify check the printed parameters to learn. When fine-tuning, this list should be long and include all the
        model parameters. However, when feature extracting this list should be short and only include the weights and
        biases of the reshaped layer.
        :param model:
        :return:
        """
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params_to_update)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        else:
            print('Optimizer not recognized')
            sys.exit()
        return optimizer
