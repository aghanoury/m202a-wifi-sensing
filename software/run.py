import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from util import load_data_n_model
from scipy import signal
from collections import defaultdict
from sklearn import metrics

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

    label_acc = torch.tensor([])
    predicted_acc = torch.tensor([])
    label_acc = label_acc.to(device)
    predicted_acc = predicted_acc.to(device)

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

        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)

        label_acc = torch.cat((label_acc, labels.to(device)), dim=0)
        predicted_acc = torch.cat((predicted_acc, predict_y.to(device)), dim=0)

    cm = metrics.confusion_matrix(label_acc.cpu(), predicted_acc.cpu())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return cm

    
def main():
    root = '/data/manny/Data/' 
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'])
    parser.add_argument('--preload', help="Path to the model to be loaded. If provided, skips training")
    parser.add_argument('--save_model', action='store_true', help="Path to save the model")
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.preload:
        torch.load(args.preload)
    else: 
        train(
            model=model,
            tensor_loader= train_loader,
            num_epochs= train_epoch,
            learning_rate=1e-3,
            criterion=criterion,
            device=device)

        if args.save_model:
            torch.save(model.state_dict(), root + 'models/{}_{}'.format(args.dataset, args.model))


    cm = test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )

    print("Confusion Matrix")
    print(cm)
    plt.figure(figsize=(6,6))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
    cm_display.plot()
    cm_name = 'results/confusion_matrix_{}_{}_{}.pdf'.format(args.dataset, args.model, train_epoch)
    plt.title('Epoch = {}, Dataset = {}'.format(train_epoch, args.dataset), pad=20)
    plt.savefig(cm_name)
    # set the title of the plot to the epoch number and dataset

    print("Saving confusion matrix to {}".format(cm_name))
    print("Name format is as follows: confusion_matrix_{dataset_model}_{epoch}.pdf")


    ## DEPRECATE 

    # # TODO: get the appropriate class names
    # if args.dataset == 'NTU-Fi_HAR':
    #     category_map = test_loader.dataset.category
    #     # swap keys and items in category_map
    #     category_map = {v: k for k, v in category_map.items()}
    #     class_names = [category_map[i] for i in range(len(category_map))]
    # else:
    #     class_names = ['1','2','3','4','5','6']
    return


if __name__ == "__main__":
    main()
