import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

"""    
Python: V 3.7.1+ 
"""

# To see available NVIDIA GPU's type "nvidia-smi"
# To select certain GPU's type "export CUDA_VISIBLE_DEVICES= <Device numbers>"
# Example "export CUDA_VISIBLE_DEVICES=1,2,5,6,7"
# CUDA Specification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(trainSet, validationSet, model, batch_size, num_epochs,
          optim_lr, optim_eps, optim_weight_decay, betas):
    """

    :param trainSet:
    :param validationSet:
    :param model:
    :param batch_size:
    :param num_epochs:
    :param optim_lr:
    :param optim_eps:
    :param optim_weight_decay:
    :param betas:
    :return:
    """
    # Create the data loader
    train_loader = DataLoader(dataset=trainSet,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    validationAcc = dict()

    criteron = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=optim_lr, betas=betas, eps=optim_eps,
                             weight_decay=optim_weight_decay, params=model.parameters())

    for epoch in range(num_epochs):
        start = time.clock()
        sumLoss = 0
        print("Epoch:" + str(epoch + 1))
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            out = model.forward(inputs)

            loss = criteron(out, labels)
            sumLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("\rLoss: " + str(sumLoss / (i + 1)), flush=True, end='')
        stop = time.clock()
        print("\nEpoch Time: " + "{:.2f}".format(stop - start) + " seconds")
        accuracy = validation(validationSet, model, batch_size)
        validationAcc['Epoch ' + str(epoch+1)] = accuracy
        print("")

    return validationAcc


def validation(validationSet, Model, batch_size):
    """

    :param validationSet:
    :param Model:
    :param batch_size:
    :return:
    """
    Valid_Loader = DataLoader(dataset=validationSet,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    total = 0
    correct = 0
    with torch.no_grad():
        for data in Valid_Loader:
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            output = Model.forward(inputs)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
    accuracy = round((100 * correct / total), 4)
    print('Validation Accuracy: {}%'.format(accuracy))
    return accuracy


def test(testSet, Model, batch_size, num_classes):
    """

    :param testSet:
    :param Model:
    :param batch_size:
    :param num_classes:
    :return:
    """

    # Create the data loader
    test_loader = DataLoader(dataset=testSet,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    classes = list("Website " + str(i) for i in range(1, num_classes+1))
    classTestAccuracy = dict()

    # Calculating metrics for the whole dataset
    correct = 0  # the amount the nn got correct
    total = 0  # Total amount of images

    with torch.no_grad():
        # Iterate through all the test set
        for data in test_loader:
            # Formatting the data
            inputs, labels = data  # separate inputs and labels

            # Grab the inputs and labels as python Variables and send them to the GPU/CPU
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            # forward pass through the model
            outputs = Model.forward(inputs)

            # Analyzing the results
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
    print('Overall Test Accuracy: {}%'.format(round((100 * correct / total), 4)))


    #  Calculating metrics for each class
    class_correct = list(0. for i in range(num_classes))  # list 0-9 all values are 0
    class_total = list(0. for i in range(num_classes))  # list 0-9 all values are 0

    # Disabling gradient calculation
    with torch.no_grad():
        #  Iterate through all of the test set
        for data in test_loader:
            inputs, labels = data  # Formatting the data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            #  Create a nn obj and pass the inputs through it
            outputs = Model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Remove single-dimensional entries from the shape of an array.
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        testAcc = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], testAcc))
        classTestAccuracy['Class' + str(i)] = testAcc

    return round(correct / total, 6), classTestAccuracy
