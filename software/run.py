import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from util import load_data_n_model
from scipy import signal
from collections import defaultdict

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
    return


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0

    # create an accuracy map, later used for the confusion matrix
    accuracy_map = defaultdict(list)

    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        for i,j in zip(labels,outputs):
            nt = torch.zeros_like(j)
            nt[torch.argmax(j)] = 1
            accuracy_map[i.item()].append(nt)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)

    confusion_matrix = np.zeros((len(accuracy_map), len(accuracy_map)))
    for k in accuracy_map:
        accuracy_map[k] = torch.stack(accuracy_map[k])
        accuracy_map[k] = torch.sum(accuracy_map[k], dim=0)
        accuracy_map[k] = accuracy_map[k].cpu().numpy()
        accuracy_map[k] = accuracy_map[k] / np.sum(accuracy_map[k])
        confusion_matrix[k] = accuracy_map[k]

    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return confusion_matrix

    
def main():
    root = '/data/manny/Data/' 
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'])
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model=model,
        tensor_loader= train_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
         )
    cm = test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )

    # plot the confusion matrix
    fig, ax = plt.subplots()
    plt.figure(figsize=(6,6))

    # does not work with UTHAR

    # TODO: get the appropriate class names
    if args.dataset == 'NTU-Fi_HAR':
        category_map = test_loader.dataset.category
        class_names = [category_map[i] for i in range(len(category_map))]
    else:
        class_name = ['1','2','3','4','5','6']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Epoch = {}'.format(train_epoch), pad=20)
    plt.colorbar()

    # TODO: this will probably cause errors for other data sets
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names, va='center')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # TODO: make the name more specific
    plt.savefig('results/confusion_matrix.pdf')

    return


if __name__ == "__main__":
    main()
