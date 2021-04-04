import os
import numpy as np
import pandas as pd
from data_prep import partition_data
from transcript_feature_extraction import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# CITATION: Some of these function are taken from my own (Charle's Reinertson) EECS 445 Project 2. 



def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # TODO: complete the training step
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    #

def _evaluate_epoch(axes, tr_loader, test_loader, model, criterion, epoch, stats):
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = torch.argmax(torch.nn.functional.softmax(output.data, dim=1), dim=1)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in test_loader:
            output = model(X)
            predicted = torch.argmax(torch.nn.functional.softmax(output.data, dim=1), dim=1)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        val_loss = np.mean(running_loss)
        val_acc = correct / total
    
    stats.append([val_acc, val_loss, train_acc, train_loss])
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(val_loss))
    print('\tValidation Accuracy: {}'.format(val_acc))
    print('\tTrain Loss: {}'.format(train_loss))
    print('\tTrain Accuracy: {}'.format(train_acc))

    valid_acc = [s[0] for s in stats]
    valid_loss = [s[1] for s in stats]
    train_acc = [s[2] for s in stats]
    train_loss = [s[3] for s in stats]
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), valid_acc,
        linestyle='--', marker='o', color='b')
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), train_acc,
        linestyle='--', marker='o', color='r')
    axes[0].legend(['Validation', 'Train'])
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes[1].legend(['Validation', 'Train'])
    plt.pause(0.00001)



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, features):
        'Initialization'
        self.labels = labels
        self.feature = features
        self.list_IDs = list_IDs


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.feature[ID]
        y = self.labels[ID]

        return X, y

class NNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 54275 output: 10000
        self.fc1 = nn.Linear(26196, 1000)
        self.drop1 = nn.Dropout(0.4)
        # input: 10000 output: 1024
        self.fc2 = nn.Linear(1000, 100)
        self.drop2 = nn.Dropout(0.4)
        # input: 64 output: 11
        self.fc3 = nn.Linear(100, 11)
        #

        self.init_weights()

    def init_weights(self):
        # initialize the parameters for [self.fc1, self.fc2, self.fc3, self.fc4]
        for fc in [self.fc1, self.fc2, self.fc3]:
            F_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(F_in))
            nn.init.constant_(fc.bias, 0.0)
        

    def forward(self, x):
        x = self.fc1(x.float())
        x = self.drop1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = F.silu(x)
        x = self.fc3(x)

        return x









def main():
    # DELETE. JUST FOR FAKE EXAMPLE DATA
    #_______________________________________________________________________________________________________________________
    # amazon = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt', delimiter = "\t", quoting=3, header=None)
    # amazon.columns = ['text', 'label']
    # transcript_features = process_transcripts(amazon.text, amazon.text)

    # categories = np.random.randint(11, size=transcript_features.shape[0]) 

    # df = pd.DataFrame({'categories': categories, 'transcript_features': list(transcript_features)}, columns=['categories', 'transcript_features'])
    # print(df.head())

    # train, test = train_test_split(df, test_size=0.2)
    # train = train.reset_index(drop=True)
    # test = test.reset_index(drop=True)
    #_______________________________________________________________________________________________________________________
    # DELETE ALL ABOVE


    
    test_file_location = "./data/test_dataframe.pkl"
    train_file_location = "./data/train_dataframe.pkl"
    validate_file_location = "./data/validate_dataframe.pkl"

  

    test = pd.read_pickle(test_file_location)
    train = pd.read_pickle(train_file_location)
    validate = pd.read_pickle(validate_file_location)

    # change indexes so right after train the tests indexes start, and right after the test indexes the validate indexes start
    test.index = np.arange(train.shape[0], train.shape[0] + test.shape[0])
    validate.index = np.arange(train.shape[0] + test.shape[0], train.shape[0] + test.shape[0] + validate.shape[0])

    # stack train, test, and validate on top one another in that exact order
    df = pd.concat([train, test, validate])
    # df = pd.concat([train, test])
    

    max_epochs = 30
    partition = {}
    labels = {}
    partition['train'] = train.index
    partition['test'] = test.index

    for index in train.index:
        labels[index] = int(train[train.index == index]['categories'])
    for index in test.index:
        labels[index] = int(test[test.index == index]['categories'])
    # for index in validate.index:
    #     labels[index] = int(validate[validate.index == index]['categories'])



    # Generators
    training_set = Dataset(partition['train'], labels, df['transcript_features'])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)

    test_set = Dataset(partition['test'], labels, test['transcript_features'])
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    # Loop over epochs


    

    # define model, loss function, and optimizer
    model = NNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    plt.ion()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    plt.suptitle(' Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    stats = []
    # Evaluate model
    _evaluate_epoch(axes, training_generator, test_generator, model.eval(), criterion, 0, stats)


    # Loop over the entire dataset multiple times
    for epoch in range(0, max_epochs):
        # Train model
        _train_epoch(training_generator, model.train(), criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, training_generator, test_generator, model.eval(), criterion, epoch+1,
            stats)


    print('Finished Training')

    fig.savefig('_training_plot.png', dpi=200)
    plt.ioff()
    plt.show()








    
    


if __name__ == "__main__":
    main()
