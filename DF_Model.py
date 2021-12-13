import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init

"""
Python: V 3.7.1+ 
"""

# To see available NVIDIA GPU's type "nvidia-smi" into terminal
# To select certain GPU's type "export CUDA_VISIBLE_DEVICES= <Device numbers>"
# Example "export CUDA_VISIBLE_DEVICES=1,2,5,6,7"

# CUDA Specification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class model(torch.nn.Module):
    def __init__(self, output_channels, kernel_size, stride, pool_kernel,
                 pool_stride, dropout_p, dropout_p_fc1, dropout_p_fc2, fc_features,
                 num_classes, input_size, dropout_inplace):
        """

        :param output_channels:
        :param kernel_size:
        :param stride:
        :param pool_kernel:
        :param pool_stride:
        :param dropout_p:
        :param dropout_p_fc1:
        :param dropout_p_fc2:
        :param fc_features:
        :param num_classes:
        :param input_size:
        :param dropout_inplace:
        """
        super().__init__()

        self.encoder = nn.Sequential(
            BasicBlock1(output_channels* 1, kernel_size, stride, pool_kernel, pool_stride, dropout_p, dropout_inplace),
            BasicBlock(output_channels * 2, kernel_size, stride, pool_kernel, pool_stride, dropout_p, dropout_inplace),
            BasicBlock(output_channels * 4, kernel_size, stride, pool_kernel, pool_stride, dropout_p, dropout_inplace),
            BasicBlock(output_channels * 8, kernel_size, stride, pool_kernel, pool_stride, dropout_p, dropout_inplace)
        )

        # The reason to do this is due to the dynamic hyperparameter tuning ability

        # Create a random tensor of the input size taken from the data
        x = Variable(torch.ones(*input_size))
        # Implement my own custom forward function
        for module in self.encoder._modules.values():
            for inner in module._modules.values():
                if isinstance(inner, nn.Conv1d) or isinstance(inner, nn.MaxPool1d):
                    # Dynamic padding that keeps 'same' padding
                    x = nn.functional.pad(x, [3, 4])
                x = inner(x)

        # Set the flatten features to the output of the fake forward pass
        self.flatten_features = int(np.prod(x.size()))
    
        # Initialize the weights of the linear layers to be 
        self.Linear1 = nn.Linear(self.flatten_features, fc_features)
        # init.xavier_normal_(self.Linear1.weight)
        self.Linear2 = nn.Linear(fc_features, fc_features)
        # init.xavier_normal_(self.Linear2.weight)
        self.Linear3 = nn.Linear(fc_features, num_classes)
        # init.xavier_normal_(self.Linear3.weight)
        
        self.decoder = nn.Sequential(
            self.Linear1,
            nn.BatchNorm1d(fc_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p_fc1, inplace=dropout_inplace),
            self.Linear2,
            nn.BatchNorm1d(fc_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p_fc2, inplace=dropout_inplace),
            self.Linear3,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # print()
        # print(x.shape)
        for module in self.encoder._modules.values():
            for inner in module._modules.values():
                # print(inner)
                # print("before: ", x.shape)
                if isinstance(inner, nn.Conv1d) or isinstance(inner, nn.MaxPool1d):
                    x = nn.functional.pad(x, [3, 4])
                x = inner(x)
            #     print("after: ", x.shape)
            #
            # print("******************************************************")

        # print(x.shape)
        x = x.view(x.size(0), -1)  # flatten
        # self.fc_channel_input = x.shape[1]
        x = self.decoder(x)
        return x


def BasicBlock1(output_channels, kernel_size, stride, pool_kernel, pool_stride, dropout_p, dropout_inplace):
    """

    :param output_channels:
    :param kernel_size:
    :param stride:
    :param pool_kernel:
    :param pool_stride:
    :param dropout_p:
    :param dropout_inplace:
    :return:
    """
    return nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=output_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm1d(output_channels),
        nn.ELU(alpha=1.0),
        nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm1d(output_channels),
        nn.ELU(alpha=1.0),
        nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride),
        nn.Dropout(p=dropout_p, inplace=dropout_inplace)
    )


def BasicBlock(output_channels, kernel_size, stride, pool_kernel, pool_stride, dropout_p, dropout_inplace):
    """

    :param output_channels:
    :param kernel_size:
    :param stride:
    :param pool_kernel:
    :param pool_stride:
    :param dropout_p:
    :param dropout_inplace:
    :return:
    """
    return nn.Sequential(
        nn.Conv1d(in_channels=output_channels // 2, out_channels=output_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm1d(output_channels),
        nn.ReLU(),
        nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm1d(output_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride),
        nn.Dropout(p=dropout_p, inplace=dropout_inplace)
    )
