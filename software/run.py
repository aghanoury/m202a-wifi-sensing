import argparse
import yaml
import json
import multiprocessing
from itertools import product
import numpy as np
import torch
import torch.nn as nn
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
    parser.add_argument('--param_run', help="Path to the yaml file containing the run parameters")
    args = parser.parse_args()

    run_params = None
    run_permutations = None
    if args.param_run:
        print("Loading parameters from {}".format(args.param_run))
        with open(args.param_run, "r") as f:
            run_params = yaml.safe_load(f)

        # parse params
        print("DEBUG")
        print(json.dumps(run_params, indent=4, sort_keys=True))

    # execute runs sequentially, execute trainings in parallel
    for run,value in run_params.items():
        if run == 'example_run': continue
        models = value['models']
        datasets = value['datasets']
        epochs = value['epochs']
        sampling_rates = value['sampling_rates']

        # create a list of all the permutations of the models and datasets
        run_permutations = list(product(models, datasets, epochs, sampling_rates))

    # for now, execute the runs sequentially
    for permutation in run_permutations:
        model_name = permutation[0]
        dataset = permutation[1]
        epoch = permutation[2]
        sampling_rate = permutation[3]

        print("DEBUG: Running permutation: {}".format(permutation))
    # testing code for multiprocessing to dispatch everything at once
    # def square(x):
    #     return x**2
    # # Create a multiprocessing Pool with 3 processes
    # with multiprocessing.Pool(processes=3) as pool:
    #     # Use apply_async to dispatch tasks asynchronously
    #     results = [pool.apply_async(square, (num,)) for num in numbers]
    #     for r in results:
    #         print(r.get())

        # before we load the data
        train_loader, test_loader, model, train_epoch = load_data_n_model(dataset, model_name, root)
        train_epoch = epoch # override epoch
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

        # print("Confusion Matrix")
        # print(cm)
        plt.figure(figsize=(6,6))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        cm_name = 'results/confusion_matrix_{}_{}_{}_SR-{}.pdf'.format(dataset, model_name, train_epoch, sampling_rate)
        plt.title('Epoch = {}, Dataset = {}'.format(train_epoch, dataset), pad=20)
        plt.savefig(cm_name)
        # set the title of the plot to the epoch number and dataset

        print("Saving confusion matrix to {}".format(cm_name))

    # if args.preload:
    #     torch.load(args.preload)
    # else: 
    #     train(
    #         model=model,
    #         tensor_loader= train_loader,
    #         num_epochs= train_epoch,
    #         learning_rate=1e-3,
    #         criterion=criterion,
    #         device=device)

    #     if args.save_model:
    #         torch.save(model.state_dict(), root + 'models/{}_{}'.format(args.dataset, args.model))


    # cm = test(
    #     model=model,
    #     tensor_loader=test_loader,
    #     criterion=criterion,
    #     device= device
    #     )


    # save results in form of confusion matrix
    

    


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
