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

def _evaluate_epoch(axes, tr_loader, validate_loader, model, criterion, epoch, stats):
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
        for X, y in validate_loader:
            output = model(X)
            predicted = torch.argmax(torch.nn.functional.softmax(output.data, dim=1), dim=1)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        val_loss = np.mean(running_loss)
        val_acc = correct / total
    
    

    stats.append([val_acc, val_loss])
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(val_loss))
    print('\tValidation Accuracy: {}'.format(val_acc))
    print('\tTrain Loss: {}'.format(train_loss))
    print('\tTrain Accuracy: {}'.format(train_acc))

    valid_acc = [s[0] for s in stats]
    valid_loss = [s[1] for s in stats]
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), valid_acc,
        linestyle='--', marker='o', color='b')
    axes[0].legend(['Validation'])
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes[1].legend(['Validation'])
    plt.pause(0.00001)

    # Uncomment if I want to see the training loss/accuracy as well

    # stats.append([val_acc, val_loss, train_acc, train_loss])
    # print('Epoch {}'.format(epoch))
    # print('\tValidation Loss: {}'.format(val_loss))
    # print('\tValidation Accuracy: {}'.format(val_acc))
    # print('\tTrain Loss: {}'.format(train_loss))
    # print('\tTrain Accuracy: {}'.format(train_acc))

    # valid_acc = [s[0] for s in stats]
    # valid_loss = [s[1] for s in stats]
    # train_acc = [s[2] for s in stats]
    # train_loss = [s[3] for s in stats]
    # axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), valid_acc,
    #     linestyle='--', marker='o', color='b')
    # axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), train_acc,
    #     linestyle='--', marker='o', color='r')
    # axes[0].legend(['Validation', 'Train'])
    # axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
    #     linestyle='--', marker='o', color='b')
    # axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
    #     linestyle='--', marker='o', color='r')
    # axes[1].legend(['Validation', 'Train'])
    # plt.pause(0.00001)



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
        # input: 29404 output: 10000
        self.fc1 = nn.Linear(29404, 10000)
        self.drop1 = nn.Dropout(0.4)
        # input: 10000 output: 1000
        self.fc2 = nn.Linear(10000, 1000)
        self.drop2 = nn.Dropout(0.4)
        # input: 1000 output: 11
        self.fc3 = nn.Linear(1000, 11)
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
    
    test_file_location = "./data/test_dataframe.pkl"
    train_file_location = "./data/train_dataframe.pkl"
    validate_file_location = "./data/validate_dataframe.pkl"

  

    train = pd.read_pickle(train_file_location)
    test = pd.read_pickle(test_file_location)
    validate = pd.read_pickle(validate_file_location)

    # change indexes so right after train the validate indexes start, and right after the validate indexes the test indexes start
    validate.index = np.arange(train.shape[0], train.shape[0] + validate.shape[0])
    test.index = np.arange(train.shape[0] + validate.shape[0], train.shape[0] + validate.shape[0] + test.shape[0])

    # stack train, validate, and test on top one another in that exact order
    df = pd.concat([train, validate, test])
    

    max_epochs = 9
    partition = {}
    labels = {}
    partition['train'] = train.index
    partition['validate'] = validate.index
    partition['test'] = test.index

    for index in train.index:
        labels[index] = int(train[train.index == index]['categories'])
    for index in validate.index:
        labels[index] = int(validate[validate.index == index]['categories'])
    for index in test.index:
        labels[index] = int(test[test.index == index]['categories'])



    # Generators
    training_set = Dataset(partition['train'], labels, df['transcript_features'])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)

    validate_set = Dataset(partition['validate'], labels, df['transcript_features'])
    validate_generator = torch.utils.data.DataLoader(validate_set, batch_size=64, shuffle=True)

    test_set = Dataset(partition['test'], labels, df['transcript_features'])
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    

    # define model, loss function, and optimizer
    model = NNetwork()
    # label numbers corresponding to the actual count in train 
    # 0: 2375,
    # 1: 2336,
    # 2: 2920,
    # 3: 2386,
    # 4: 2310,
    # 5: 1704,
    # 6: 2181,
    # 7: 2106,
    # 8: 4956,
    # 9: 1788,
    # 10: 2343,
    weights = torch.tensor([2375, 2336, 2920, 2386, 2310, 1704, 2181, 2106, 4956, 1788, 2343], dtype=torch.float32)
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
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
    _evaluate_epoch(axes, training_generator, validate_generator, model.eval(), criterion, 0, stats)


    # Loop over the entire dataset multiple times
    for epoch in range(0, max_epochs):
        # Train model
        _train_epoch(training_generator, model.train(), criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, training_generator, validate_generator, model.eval(), criterion, epoch+1,
            stats)


    print('Finished Training')

    # label numbers corresponding to the actual category names
    # 0:'business',
    # 1:'comedy',
    # 2:'education',
    # 3:'games',
    # 4:'health',
    # 5:'music',
    # 6:'politics',
    # 7:'society',
    # 8:'spirituality',
    # 9:'sports',
    # 10:'technology',
    
    category_to_accuracy = {
        'business':0,
        'comedy':0,
        'education':0,
        'games':0,
        'health':0,
        'music':0,
        'politics':0,
        'society':0,
        'spirituality':0,
        'sports':0,
        'technology':0,
    }
    # label numbers to count of correct labels as well as total labels
    index_to_count = {
        'correct0':0,
        'total0':0,
        'correct1':0,
        'total1':0,
        'correct2':0,
        'total2':0,
        'correct3':0,
        'total3':0,
        'correct4':0,
        'total4':0,
        'correct5':0,
        'total5':0,
        'correct6':0,
        'total6':0,
        'correct7':0,
        'total7':0,
        'correct8':0,
        'total8':0,
        'correct9':0,
        'total9':0,
        'correct10':0,
        'total10':0,
    }

    # Calculate test accuracy and loss as well as accuracy for each category
    model = model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in test_generator:
                output = model(X)
                predicted = torch.argmax(torch.nn.functional.softmax(output.data, dim=1), dim=1)
                y_true.append(y)
                y_pred.append(predicted)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())
                # predict accuracy per category
                for i in range(0, 11):
                    pred_actual = (predicted == y)
                    pred_num = (y == i)
                    holder = torch.where(pred_num == True, 1, 5)
                    to_sum = (pred_actual == holder)
          
                    index_to_count[('correct' + str(i))] += (to_sum).sum().item()
                    index_to_count[('total' + str(i))] += (y == i).sum().item()

        test_loss = np.mean(running_loss)
        test_acc = correct / total
        category_to_accuracy['business'] = index_to_count['correct0'] / index_to_count['total0']
        category_to_accuracy['comedy'] = index_to_count['correct1'] / index_to_count['total1']
        category_to_accuracy['education'] = index_to_count['correct2'] / index_to_count['total2']
        category_to_accuracy['games'] = index_to_count['correct3'] / index_to_count['total3']
        category_to_accuracy['health'] = index_to_count['correct4'] / index_to_count['total4']
        category_to_accuracy['music'] = index_to_count['correct5'] / index_to_count['total5']
        category_to_accuracy['politics'] = index_to_count['correct6'] / index_to_count['total6']
        category_to_accuracy['society'] = index_to_count['correct7'] / index_to_count['total7']
        category_to_accuracy['spirituality'] = index_to_count['correct8'] / index_to_count['total8']
        category_to_accuracy['sports'] = index_to_count['correct9'] / index_to_count['total9']
        category_to_accuracy['technology'] = index_to_count['correct10'] / index_to_count['total10']
    
    print("Test loss: {0}".format(test_loss))
    print("Test recall: {0}".format(test_acc))
    print("Test category business recall: {0}".format(category_to_accuracy['business']))
    print("Test category comedy recall: {0}".format(category_to_accuracy['comedy']))
    print("Test category education recall: {0}".format(category_to_accuracy['education']))
    print("Test category games recall: {0}".format(category_to_accuracy['games']))
    print("Test category health recall: {0}".format(category_to_accuracy['health']))
    print("Test category music recall: {0}".format(category_to_accuracy['music']))
    print("Test category politics recall: {0}".format(category_to_accuracy['politics']))
    print("Test category society recall: {0}".format(category_to_accuracy['society']))
    print("Test category spirituality recall: {0}".format(category_to_accuracy['spirituality']))
    print("Test category sports recall: {0}".format(category_to_accuracy['sports']))
    print("Test category technology recall: {0}".format(category_to_accuracy['technology']))


    with open("test_bag_of_words_features.txt", "w+") as text_file:
        text_file.write("Test loss: {0} \n".format(test_loss))
        text_file.write("Test accuracy: {0} \n".format(test_acc))
        text_file.write("Test category business recall: {0} \n".format(category_to_accuracy['business']))
        text_file.write("Test category comedy recall: {0} \n".format(category_to_accuracy['comedy']))
        text_file.write("Test category education recall: {0} \n".format(category_to_accuracy['education']))
        text_file.write("Test category games recall: {0} \n".format(category_to_accuracy['games']))
        text_file.write("Test category health recall: {0} \n".format(category_to_accuracy['health']))
        text_file.write("Test category music recall: {0} \n".format(category_to_accuracy['music']))
        text_file.write("Test category politics recall: {0} \n".format(category_to_accuracy['politics']))
        text_file.write("Test category society recall: {0} \n".format(category_to_accuracy['society']))
        text_file.write("Test category spirituality recall: {0} \n".format(category_to_accuracy['spirituality']))
        text_file.write("Test category sports recall: {0} \n".format(category_to_accuracy['sports']))
        text_file.write("Test category technology recall: {0} \n".format(category_to_accuracy['technology']))


    # save stats as numpy array to csv file
    stats = np.array(stats)
    np.savetxt('stats_bag_of_words_features.csv', stats, delimiter=',')

    fig.savefig('_training_plot.png', dpi=200)
    plt.ioff()
    plt.show()








    
    


if __name__ == "__main__":
    main()
