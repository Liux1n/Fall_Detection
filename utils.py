import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import zipfile
import tempfile


def truth_test(_test,_pred, i):
    _test = np.array(_test)
    _pred = np.array(_pred)
    
    _pred_pos = _test[_pred == i]
    _pred_neg = _test[_pred != i]
    
    _true_pos = len(_pred_pos[_pred_pos == i])
    _fals_pos = len(_pred_pos[_pred_pos != i])
    
    _true_neg = len(_pred_neg[_pred_neg != i])
    _fals_neg = len(_pred_neg[_pred_neg == i])
    
    return _true_pos, _fals_pos, _true_neg, _fals_neg

def sensitivity(_test,_pred, i):
    tp, fp, tn, fn = truth_test(_test, _pred, i)
    return tp / ( tp + fn)

def specificity(_test,_pred, i):
    tp, fp, tn, fn = truth_test(_test, _pred, i)
    return tn / ( tn + fp)

def accuracy(_test, _pred, i):
    tp, fp, tn, fn = truth_test(_test, _pred, i)
    return (tp+tn) / (tp + fp + tn + fn)


def train(
        dataloader, model, loss_fn, optimizer, val_dataloader, 
        patience=5, scheduler=None, device = None, epochs = None, B_size = 1, A_size = 1
        ):
    
    size = len(dataloader.dataset)
    device = device
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch}:")
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            model = model.to(device)
            B_multiplier = 1
            A_multiplier = B_size / A_size
            multipliers = torch.where(y == 0, B_multiplier, A_multiplier)
            pred = model(X.float())
            loss = (loss_fn(pred, y) * multipliers).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        val_loss = val_loop(val_dataloader, model, loss_fn, device = device)

        # Step the scheduler with the validation loss
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break

def val_loop(dataloader, model, loss_fn, device = None):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0
    device = device
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #model = model.to(device)
            pred = model(X.float())
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    
    return val_loss

def test(dataloader, model, loss_fn,  device = None):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    y_true = []
    y_pred = []
    device = device
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            model = model.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_true.extend(y.tolist())
            y_pred.extend(pred.argmax(1).tolist())

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Calculate accuracy, specificity, and sensitivity for each class
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Specificity = TN / (TN + FP)
    # Sensitivity = TP / (TP + FN)
    for i in range(2):  # Adjust the range depending on the number of classes
        print(" Label", i)
        print("    accuracy\t%5.3f"%accuracy(y_true, y_pred, i))
        print(" specificity\t%5.3f"%specificity(y_true, y_pred, i))
        print(" sensitivity\t%5.3f"%sensitivity(y_true, y_pred, i))

    # create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    # show the confusion matrix on a plot
    plt.matshow(cm)
    # add legend
    plt.colorbar()
    # add title
    plt.title('Confusion Matrix (0: not fall, 1: fall)')




def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    

def get_gzipped_model_size(file):
    # It returns the size of the gzipped model in bytes.
    
    
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    
    return os.path.getsize(zipped_file)