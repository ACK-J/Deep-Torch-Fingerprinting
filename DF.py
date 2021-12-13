import pickle, torch, time, argparse, os, datetime, json
import numpy as np
from torch.utils.data import Dataset
from pprint import pprint
import torch.nn as nn
import DF_train_test_valid
from DF_Model import model


""" 
Python: V 3.7.1+ 

TODO Make sure that the output channels pad correctly for even and odd padding
TODO Test out other optimizers
"""

# To see available NVIDIA GPU's type "nvidia-smi"
# To select certain GPU's type "export CUDA_VISIBLE_DEVICES= <Device numbers>"
# Example "export CUDA_VISIBLE_DEVICES=1,2,5,6,7"
# CUDA Specification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):
    """
    Class to hold the X and y data which will then be put into a dataloader
    """
    def __init__(self,  x_train, y_train):
        self.len = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    """
    Class to hold the X and y data which will then be put into a dataloader
    """
    def __init__(self,  x_test, y_test):
        self.len = x_test.shape[0]
        self.x_test = x_test
        self.y_test = y_test

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]

    def __len__(self):
        return self.len


class ValidDataset(Dataset):
    """
    Class to hold the X and y data which will then be put into a dataloader
    """
    def __init__(self,  x_valid, y_valid):
        self.len = x_valid.shape[0]
        self.x_valid = x_valid
        self.y_valid = y_valid

    def __getitem__(self, index):
        return self.x_valid[index], self.y_valid[index]

    def __len__(self):
        return self.len

def ascii():
    print()
    print("__/\\\\\\\\\\\\\\\\\\\\\\\\_____/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_       ")
    print(" _\\/\\\\\\////////\\\\\\__\\/\\\\\\///////////__       ")
    print("  _\\/\\\\\\______\\//\\\\\\_\\/\\\\\\_____________     ")
    print("   _\\/\\\\\\_______\\/\\\\\\_\\/\\\\\\\\\\\\\\\\\\\\\\_____    ")
    print("    _\\/\\\\\\_______\\/\\\\\\_\\/\\\\\\///////______  ")
    print("     _\\/\\\\\\_______\\/\\\\\\_\\/\\\\\\_____________")
    print("      _\\/\\\\\\_______/\\\\\\__\\/\\\\\\_____________ ")
    print("       _\\/\\\\\\\\\\\\\\\\\\\\\\\\/___\\/\\\\\\_____________")
    print("        _\\////////////_____\\///______________")


def unpickle(DATA_DIR):
    """

    :param DATA_DIR:
    :return:
    """
    print("Loading the data...")
    t1 = time.perf_counter()

    # unpickle the training data and store it in a numpy array
    with open(DATA_DIR + "X_train_NoDef.pkl", "rb") as handle:
        X_train_NoDef = np.array(pickle.load(handle, encoding='bytes'))
    with open(DATA_DIR + "y_train_NoDef.pkl", "rb") as handle:
        y_train_NoDef = np.array(pickle.load(handle, encoding='bytes'))

    # unpickle the validation data and store it in a numpy array
    with open(DATA_DIR + "X_valid_NoDef.pkl", "rb") as handle:
        X_valid_NoDef = np.array(pickle.load(handle, encoding='bytes'))
    with open(DATA_DIR + "y_valid_NoDef.pkl", "rb") as handle:
        y_valid_NoDef = np.array(pickle.load(handle, encoding='bytes'))

    # unpickle the testing data and store it in a numpy array
    with open(DATA_DIR + "X_test_NoDef.pkl", "rb") as handle:
        X_test_NoDef = np.array(pickle.load(handle, encoding='bytes'))
    with open(DATA_DIR + "y_test_NoDef.pkl", "rb") as handle:
        y_test_NoDef = np.array(pickle.load(handle, encoding='bytes'))

    # Print out to the user, how long loading the data took
    t2 = time.perf_counter()
    print("Data Loaded in " + str('%.2f'%(t2 - t1)) + " seconds!")

    return X_train_NoDef, y_train_NoDef, X_valid_NoDef, y_valid_NoDef, X_test_NoDef, y_test_NoDef


def Parse():
    """

    :return:
    """

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", dest="epochs", default=30,
                        type=int, help="1 epoch = 1 forward pass and 1 backward pass of all the training examples [INT]")
    parser.add_argument("-o", "--outputChannels", dest="output_Channels", default=32,
                        type=int, help="The DF model takes in input with only 1 channel,"
                                       " this number is the amount of channels desired after "
                                       "the first convolution layer which will continuously be"
                                       " doubled through out the network [INT]")
    parser.add_argument("-lr", "--learningRate", dest="learning_rate", default=0.002,
                        type=float, help="Learning rate for the adamax optimizer [FLOAT]")
    parser.add_argument("-eps", "--epsilon", dest="epsilon", default=1E-8,
                        type=float, help="Epsilon value for the optimizer, added to "
                                         "the denominator to improve numerical stability [FLOAT]")
    parser.add_argument("-wd", "--weightDecay", dest="weight_decay", default=0.0,
                        type=float, help="float >= 0. Weight decay over each update. [FLOAT]")
    parser.add_argument("-bs", "--betas", dest="betas", default=(0.9, 0.99),
                        type=tuple, help="The exponential decay rate for the first and second moment estimates."
                                         " Tuple of floats, 0 < beta < 1. Generally close to 1. [TUPLE]")
    parser.add_argument("-b", "--batchSize", dest="batch_size", default=128,
                        type=int, help="The number of training examples in one forward/backward pass."
                                       " The higher the batch size the more memory you will need. [INT]")
    parser.add_argument("-k", "--kernelSize", dest="kernel_size", default=8,
                        type=int, help="The 1d kernel filter size. [INT]")
    parser.add_argument("-n", "--numberClasses", dest="num_classes", default=95,
                        type=int, help="The number of classes your data has. [INT]")
    parser.add_argument("-s", "--stride", dest="stride", default=1,
                        type=int, help="The stride size for the convolution layers. [INT]")
    parser.add_argument("-pk", "--poolKernel", dest="pool_kernel", default=8,
                        type=int, help="The kernel size for max-pooling layers. [INT]")
    parser.add_argument("-ps", "--poolStride", dest="pool_stride", default=4,
                        type=int, help="The stride size for max-pooling layers. [INT]")
    parser.add_argument("-dp", "--dropoutP", dest="dropout_p_value", default=0.1,
                        type=float, help="The p-value for the dropout layers in the basic blocks. [FLOAT]")
    parser.add_argument("-dpf1", "--dropoutPFC1", dest="dropout_p_fc1_value", default=0.7,
                        type=float, help="The p-value for the dropout layer after the first FC layer. [FLOAT]")
    parser.add_argument("-dpf2", "--dropoutPFC2", dest="dropout_p_fc2_value", default=0.5,
                        type=float, help="The p-value for the dropout layer after the second FC layer. [FLOAT]")
    parser.add_argument("-di", "--dropoutInplace", dest="dropout_inplace", default=False,
                        type=bool, help="A boolean value for all the dropout layers. [BOOLEAN]")
    parser.add_argument("-fc", "--fullyConnected", dest="fully_connected_size", default=512,
                        type=int, help="The number of channels in the fully connected layer. [INT]")
    parser.add_argument("-v", "--verbose", dest="verbose", default=False,
                        type=bool, help="Decides if you want to produce verbose output")

    args = parser.parse_args()

    params = dict()
    params['epochs'] = int(args.epochs)
    params['outputChannels'] = int(args.output_Channels)
    params['learningRate'] = float(args.learning_rate)
    params['epsilon'] = float(args.epsilon)
    params['weightDecay'] = float(args.weight_decay)
    params['betas'] = tuple(args.betas)
    params['batchSize'] = int(args.batch_size)
    params['kernelSize'] = int(args.kernel_size)
    params['num_classes'] = int(args.num_classes)
    params['stride'] = int(args.stride)
    params['poolKernel'] = int(args.pool_kernel)
    params['poolStride'] = int(args.pool_stride)
    params['dropoutP'] = float(args.dropout_p_value)
    params['dropoutPFC1'] = float(args.dropout_p_fc1_value)
    params['dropoutPFC2'] = float(args.dropout_p_fc2_value)
    params['dropoutInplace'] = bool(args.dropout_inplace)
    params['fully_connected_size'] = int(args.fully_connected_size)
    params['verbose'] = bool(args.verbose)
    ascii()
    print()
    print('HYPER-PARAMETERS:')
    pprint(params)
    print()
    return params


if __name__ == '__main__':
    """
    
    """

    # Grab the hyper parameters from command line input
    params = Parse()
    epochs               = params['epochs']
    outputChannels       = params['outputChannels']
    learningRate         = params['learningRate']
    epsilon              = params['epsilon']
    weightDecay          = params['weightDecay']
    betas                = params['betas']
    batchSize            = params['batchSize']
    kernelSize           = params['kernelSize']
    num_classes          = params['num_classes']
    stride               = params['stride']
    poolKernel           = params['poolKernel']
    poolStride           = params['poolStride']
    dropoutP             = params['dropoutP']
    dropoutPFC1          = params['dropoutPFC1']
    dropoutPFC2          = params['dropoutPFC2']
    dropoutInplace       = params['dropoutInplace']
    fully_connected_size = params['fully_connected_size']
    verbose              = params['verbose']

    DATA_DIR = "/home/jack/PyTorch/Research/DF/NoDef/"

    FILEPATH = "/home/jack/PyTorch/Research/DF/models/"
    MODEL_NAME = "model.df"

    print("Cuda available = " + str(torch.cuda.is_available()))

    # Unpickle all of the data and return it as numpy arrays
    X_train_NoDef, y_train_NoDef, X_valid_NoDef, y_valid_NoDef, X_test_NoDef, y_test_NoDef = unpickle(DATA_DIR)
    # (76000, 5000) (76000,) (9500, 5000) (9500,) (9500, 5000) (9500,)

    print("Reshaping and processing the data...")

    inputSize = (1, 1, X_train_NoDef.shape[1])

    # Convert the X data into floating point tensors
    X_train_NoDef = torch.from_numpy(X_train_NoDef).float()  # torch.Size([76000, 5000])
    X_valid_NoDef = torch.from_numpy(X_valid_NoDef).float()  # torch.Size([9500, 5000])
    X_test_NoDef = torch.from_numpy(X_test_NoDef).float()    # torch.Size([9500, 5000])

    X_train_NoDef = X_train_NoDef.unsqueeze(2)  # torch.Size([76000, 5000, 1])
    X_valid_NoDef = X_valid_NoDef.unsqueeze(2)  # torch.Size([9500, 5000, 1])
    X_test_NoDef = X_test_NoDef.unsqueeze(2)    # torch.Size([9500, 5000, 1])

    X_train_NoDef = np.transpose(X_train_NoDef, axes=(0, 2, 1))  # torch.Size([76000, 1, 5000])
    X_valid_NoDef = np.transpose(X_valid_NoDef, axes=(0, 2, 1))  # torch.Size([9500, 1, 5000])
    X_test_NoDef = np.transpose(X_test_NoDef, axes=(0, 2, 1))    # torch.Size([9500, 1, 5000])

    # Convert the y Data into long tensors
    y_train_NoDef = torch.from_numpy(y_train_NoDef).long()   # torch.Size([76000, 95])
    y_valid_NoDef = torch.from_numpy(y_valid_NoDef).long()   # torch.Size([9500, 95])
    y_test_NoDef = torch.from_numpy(y_test_NoDef).long()     # torch.Size([9500, 95])

    # Create the datasets for the training, testing and validation set
    trainSet = TrainDataset(x_train=X_train_NoDef, y_train=y_train_NoDef)
    testSet = TestDataset(x_test=X_test_NoDef, y_test=y_test_NoDef)
    validSet = ValidDataset(x_valid=X_valid_NoDef, y_valid=y_valid_NoDef)

    if os.path.isfile(FILEPATH + MODEL_NAME):
        DF_Model = torch.load(FILEPATH + MODEL_NAME)
        if torch.cuda.device_count() > 1:
            DF_Model = nn.DataParallel(DF_Model)
        DF_Model.to(device)
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                print("Using 1 GPU!")
            else:
                print("Using", torch.cuda.device_count(), "GPUs!")
        print()

        print("Testing:")
        test_accuracy, classTestAcc = DF_train_test_valid.test(testSet, Model=model,
                                                                        batch_size=batchSize,
                                                                        num_classes=num_classes)
        test_accuracy = str(test_accuracy).split('.')[1]
        Out_File_Name = "E" + str(epochs) + ":" + "O" + str(outputChannels) + ":" + "B" + str(batchSize) + ":" + "K" \
                        + str(kernelSize) + ":" + "AC" + test_accuracy + ".df"

        Dir = FILEPATH + str(test_accuracy)
        # Checking to see if the directory already exists
        if not os.path.isdir(Dir):
            # If not then make the directory, save the model and the config
            os.makedirs(Dir)
            torch.save(DF_Model, Dir + "/" + Out_File_Name)
            with open(Dir + '/Hyper-parameters.json', 'w') as fp:
                json.dump(params, fp)
            with open(Dir + '/ClassTestAcc.json', 'w') as fp:
                json.dump(classTestAcc, fp)
        else:
            date = str(datetime.datetime.utcnow())
            os.makedirs(Dir + date)
            torch.save(DF_Model, Dir + date + "/" + Out_File_Name)
            with open(Dir + date + '/Hyper-parameters.json', 'w') as fp:
                json.dump(params, fp)
            with open(Dir + date + '/ClassTestAcc.json', 'w') as fp:
                json.dump(classTestAcc, fp)

    else:
        DF_Model = model(output_channels=outputChannels, kernel_size=kernelSize, stride=stride,
                         pool_kernel=poolKernel, pool_stride=poolStride, dropout_p=dropoutP,
                         dropout_p_fc1=dropoutPFC1, fc_features=fully_connected_size,
                         num_classes=num_classes, input_size=inputSize, dropout_p_fc2=dropoutPFC2,
                         dropout_inplace=dropoutInplace).to(device=device)

        if torch.cuda.device_count() > 1:
            DF_Model = nn.DataParallel(DF_Model)
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                print("Using 1 GPU!")
            else:
                print("Using", torch.cuda.device_count(), "GPUs!")
        print()

        print("Training:")
        validationAccDict = DF_train_test_valid.train(trainSet, validSet, model=DF_Model, batch_size=batchSize,
                                  num_epochs=epochs, optim_lr=learningRate, optim_eps=epsilon,
                                  optim_weight_decay=weightDecay, betas=betas)

        print("Testing:")
        test_accuracy, classTestAcc = DF_train_test_valid.test(testSet, Model=DF_Model, batch_size=batchSize,
                                                               num_classes=num_classes)

        test_accuracy = str(test_accuracy).split('.')[1]
        Out_File_Name = "E" + str(epochs) + ":" + "O" + str(outputChannels) + ":" + "B" + str(batchSize) + ":" + "K"\
                        + str(kernelSize) + ":" + "AC" + test_accuracy + ".df"

        Dir = FILEPATH + str(test_accuracy)
        # Checking to see if the directory already exists
        if not os.path.isdir(Dir):
            # If not then make the directory, save the model and the config
            os.makedirs(Dir)
            torch.save(DF_Model, Dir + "/" + Out_File_Name)
            with open(Dir + '/Hyper-parameters.json', 'w') as fp:
                json.dump(params, fp)
            with open(Dir + '/ValidationAcc.json', 'w') as fp:
                json.dump(validationAccDict, fp)
            with open(Dir + '/ClassTestAcc.json', 'w') as fp:
                json.dump(classTestAcc, fp)
        else:
            date = str(datetime.datetime.utcnow())
            os.makedirs(Dir + date)
            torch.save(DF_Model, Dir + date + "/" + Out_File_Name)
            with open(Dir + date + '/Hyper-parameters.json', 'w') as fp:
                json.dump(params, fp)
            with open(Dir + date + '/ValidationAcc.json', 'w') as fp:
                json.dump(validationAccDict, fp)
            with open(Dir + date + '/ClassTestAcc.json', 'w') as fp:
                json.dump(classTestAcc, fp)
    if verbose:
        print()
        print("Model Layers:")
        print(str(DF_Model))


