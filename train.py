import time
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
import sklearn


def get_loss_fn():
    return nn.BCEWithLogitsLoss()


def calculate_loss(logits, labels, loss_fn):
    loss = loss_fn(logits, labels)
    return loss


def get_optimizer(net, lr, weight_decay):
    return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


def get_hyper_parameters():
    hidden_dim = [128, 300, 512]
    lr = [1e-5, 1e-4, 1e-3]
    weight_decay = [1e-6, 1e-5, 1e-4]
    return hidden_dim, lr, weight_decay


def train_model(net, trn_loader, val_loader, optim, scheduler, num_epoch=50,
                patience=10, collect_cycle=30, device='cpu', verbose=True):
    """
    Train the model
    Input:
        - net: model
        - trn_loader: dataloader for training data
        - val_loader: dataloader for validation data
        - optim: optimizer
        - scheduler: learning rate scheduler
        - num_epoch: number of epochs to train
        - collect_cycle: how many iterations to collect training statistics
        - device: device to use
        - verbose: whether to print training details
    Return:
        - best_model: the model that has the best performance on validation data
        - stats: training statistics
    """
    # Initialize:
    # -------------------------------------
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0
    num_bad_epoch = 0

    loss_fn = get_loss_fn()
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        # Training:
        net.train()
        for sentences, labels in trn_loader:
            num_itr += 1
            ############ TODO: calculate loss, update weights ############
            sentences = [i.to(device) for i in sentences]
            labels = labels.to(device)

            optim.zero_grad()
            logits = net(sentences)

            # print(labels)
            # print(gdfs)

            # print(logits)
            # print(labels)

            loss = calculate_loss(logits, labels, loss_fn)

            loss.backward()
            optim.step()
            # for name, param in net.named_parameters():
            #   if param.requires_grad:
            #     print(name) #, param.data)
            #     print(param.grad)
            ###################### End of your code ######################

            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            # print(logits)
            # print(labels)
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
            ))

        # Validation:
        accuracy, loss = get_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation loss: {:.4f}".format(loss))
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # early stopping
        if num_bad_epoch >= patience:
            break

        # learning rate scheduler
        scheduler.step()

    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start) / 60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_accuracy,
             }

    return best_model, stats


def get_performance(net, loss_fn, data_loader, device, prediction_file=None):
    """
    Evaluate model performance on validation set or test set.
    Input:
        - net: model
        - loss_fn: loss function
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
        - prediction_file: if not None, it's filename for the file that stores predictions
    Return:
        - accuracy: accuracy on validation set
        - loss: loss on validation set
    """
    net.eval()
    y_true = []  # true labels
    y_pred = []  # predicted labels
    total_loss = []  # loss for each batch

    with torch.no_grad():
        for sentences, labels in data_loader:
            loss = None  # loss for this batch
            pred = None
            """
            pred: predicted sentence_id for each question in the batch.
            Predict -1 if the question is unanswerable.
            Use P = 0.5 as the threshold for prediction.
            A question is unanswerable if all context sentences have probability < 0.5.
            Shape: 1-d tensor of length |Q_1| + |Q_2| + ..., where |Q_i| is the
            number of questions in data point i.
            """

            ######## TODO: calculate loss, get predictions #########
            sentences = [i.to(device) for i in sentences]
            labels = labels.to(device)
            # print(labels)
            logits = net.forward(sentences)
            loss = calculate_loss(logits, labels, loss_fn)

            # print(logits)
            # print(labels)

            maximum, idx = torch.max(logits, dim=1)
            pred = idx
            # print(pred)

            ###################### End of your code ######################

            total_loss.append(loss.item())
            # y_true.append(labels[labels == 1].cpu())
            y_true.append(torch.argmax(labels, dim=1).cpu())
            
            y_pred.append(pred.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    # print(len(y_true[y_true == 0]))
    # print(len(y_true[y_true == 1]))
    # print(len(y_true[y_true == 2]))
    # print(y_pred)
    accuracy = (y_true == y_pred).sum() / y_pred.shape[0]
    total_loss = sum(total_loss) / len(total_loss)
    # save predictions
    if prediction_file is not None:
        torch.save(y_pred, prediction_file)

    return accuracy, total_loss



def evaluate(net, loss_fn, data_loader, device, prediction_file=None):
    """
    Evaluate model performance on validation set or test set.
    Input:
        - net: model
        - loss_fn: loss function
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
        - prediction_file: if not None, it's filename for the file that stores predictions
    Return:
        - accuracy: accuracy on validation set
        - loss: loss on validation set
    """
    net.eval()
    y_true = []  # true labels
    y_pred = []  # predicted labels
    total_loss = []  # loss for each batch

    with torch.no_grad():
        for sentences, labels in data_loader:
            loss = None  # loss for this batch
            pred = None
            """
            pred: predicted sentence_id for each question in the batch.
            Predict -1 if the question is unanswerable.
            Use P = 0.5 as the threshold for prediction.
            A question is unanswerable if all context sentences have probability < 0.5.
            Shape: 1-d tensor of length |Q_1| + |Q_2| + ..., where |Q_i| is the
            number of questions in data point i.
            """

            ######## TODO: calculate loss, get predictions #########
            sentences = [i.to(device) for i in sentences]
            labels = labels.to(device)
            # print(labels)
            logits = net.forward(sentences)
            loss = calculate_loss(logits, labels, loss_fn)

            # print(logits)
            # print(labels)

            maximum, idx = torch.max(logits, dim=1)
            pred = idx
            # print(pred)

            ###################### End of your code ######################

            total_loss.append(loss.item())
            # y_true.append(labels[labels == 1].cpu())
            y_true.append(torch.argmax(labels, dim=1).cpu())
            
            y_pred.append(pred.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    # print(len(y_true[y_true == 0]))
    # print(len(y_true[y_true == 1]))
    # print(len(y_true[y_true == 2]))
    # print(y_pred)
    accuracy = (y_true == y_pred).sum() / y_pred.shape[0]
    total_loss = sum(total_loss) / len(total_loss)
    # save predictions
    if prediction_file is not None:
        torch.save(y_pred, prediction_file)

    mac_f1 = sklearn.metrics.f1_score(y_pred, y_true, average='macro')
    mic_f1 = sklearn.metrics.f1_score(y_pred, y_true, average='micro')

    return accuracy.item(), total_loss, mac_f1, mic_f1


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()