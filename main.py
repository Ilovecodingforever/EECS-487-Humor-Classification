import itertools
from tqdm.notebook import tqdm
from train import get_optimizer, train_model, plot_loss, get_performance, get_hyper_parameters, get_loss_fn
from model import Model
from naive_bayes import NaiveBayes, evaluate
from data_processor import basic_collate_fn, HumorDataset, load_data, load_data_nb
import os
import torch
from torch.utils.data import DataLoader
import random


def search_param_basic(train_loader, dev_loader):
    """Experiemnt on different hyper parameters."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    hidden_dim, learning_rate, weight_decay = get_hyper_parameters()
    print("hidden dimension from: {}\nlearning rate from: {}\nweight_decay from: {}".format(
        hidden_dim, learning_rate, weight_decay
    ))
    best_model, best_stats = None, None
    best_accuracy, best_lr, best_wd, best_hd = 0, 0, 0, 0
    for hd, lr, wd in tqdm(itertools.product(hidden_dim, learning_rate, weight_decay),
                           total=len(hidden_dim) * len(learning_rate) * len(weight_decay)):
        net = Model(hd).to(device)
        optim = get_optimizer(net, lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0,
                                                      end_factor=0, total_iters=40)
        model, stats = train_model(net, train_loader, dev_loader, optim, scheduler,
                                   num_epoch=40, patience=10, collect_cycle=500,
                                   device=device, verbose=True)
        # print accuracy
        print(f"{(hd, lr, wd)}: {stats['accuracy']}")
        # update best parameters if needed
        if stats['accuracy'] > best_accuracy:
            best_accuracy = stats['accuracy']
            best_model, best_stats = model, stats
            best_hd, best_lr, best_wd = hd, lr, wd
    print("\n\nBest hidden dimension: {}, Best learning rate: {}, best weight_decay: {}".format(
        best_hd, best_lr, best_wd))
    print("Accuracy: {:.4f}".format(best_accuracy))
    plot_loss(best_stats)
    return best_model


def overfit(train_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    small_data = torch.utils.data.Subset(train_data, random.sample(range(0, len(train_data)), 500))
    small_loader = DataLoader(small_data, batch_size=16, collate_fn=basic_collate_fn, shuffle=True)

    hidden_dim = 256
    net = Model(hidden_dim).to(device)

    optim = get_optimizer(net, lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0,
                                                  end_factor=0, total_iters=50)
    best_model, stats = train_model(net, small_loader, small_loader, optim, scheduler,
                                    num_epoch=50, patience=20, collect_cycle=20, device=device)
    plot_loss(stats)


def baseline():
    """
    baseline model
    prints metrics
    """
    train = load_data_nb("train.csv")
    dev = load_data_nb("dev.csv")
    test = load_data_nb("gold-test-27446.csv")

    naive_bayes = NaiveBayes()
    naive_bayes.fit(train)

    alphas = [0.5, 0.6, 0.7]
    best_acc = 0
    best_alpha = 0
    for a in alphas:
        predictions = naive_bayes.predict(dev.text.tolist(), a)
        labels = dev.is_humor.tolist()
        accuracy, mac_f1, mic_f1 = evaluate(predictions, labels)
        if accuracy > best_acc:
            best_alpha = a
            best_acc = accuracy

    predictions = naive_bayes.predict(test.text.tolist(), best_alpha)
    labels = test.is_humor.tolist()
    accuracy, mac_f1, mic_f1 = evaluate(predictions, labels)
    print(f"Accuracy: {accuracy}")
    print(f"Macro f1: {mac_f1}")
    print(f"Micro f1: {mic_f1}")


# if __name__ == "__main__":
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    is_overfit = True
    is_hyperparam = False
    is_baseline = False

    if is_baseline:
        baseline()

    x_train, y_train = load_data("/content/drive/My Drive/EECS-487-Project/train.csv")
    x_dev, y_dev = load_data("/content/drive/My Drive/EECS-487-Project/dev.csv")
    x_test, y_test = load_data("/content/drive/My Drive/EECS-487-Project/gold-test-27446.csv")
    train_data = HumorDataset(x_train, y_train)
    dev_data = HumorDataset(x_dev, y_dev)
    test_data = HumorDataset(x_test, y_test)

    if is_overfit:
        overfit(train_data)

    train_loader = DataLoader(train_data, batch_size=2, collate_fn=basic_collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=2, collate_fn=basic_collate_fn, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, collate_fn=basic_collate_fn, shuffle=True)


    if is_hyperparam:
        model = search_param_basic(train_loader, dev_loader)
    else:
        model = Model(256)

    loss_fn = get_loss_fn()
    _, _ = get_performance(model, loss_fn, test_loader, device,
                "/content/drive/My Drive/EECS-487-Project/basic.pt")
