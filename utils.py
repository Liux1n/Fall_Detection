import numpy as np
# import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import zipfile
import tempfile
from scipy.stats import gaussian_kde
from typing import Optional 

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


def rescale(x, min, max, new_min, new_max):
    return (x - min) / (max - min) * (new_max - new_min) + new_min

def plot_CDF(fall_data, not_fall_data, title):
    # Compute KDEs
    kde_fall = gaussian_kde(fall_data.flatten())
    kde_not_fall = gaussian_kde(not_fall_data.flatten())

    # Generate values for x-axis
    x = np.linspace(min(fall_data.min(), not_fall_data.min()), 
                    max(fall_data.max(), not_fall_data.max()), 1000)

    # Compute densities
    density_fall = kde_fall(x)
    density_not_fall = kde_not_fall(x)

    # Compute CDFs
    cdf_fall = np.cumsum(density_fall) / np.sum(density_fall)
    cdf_not_fall = np.cumsum(density_not_fall) / np.sum(density_not_fall)

    # Plot CDFs
    plt.figure()
    plt.plot(x, cdf_fall, label='fall')
    plt.plot(x, cdf_not_fall, label='not fall')
    plt.legend()
    plt.title(title)
    plt.show()


def rescale_data(
    data: np.ndarray,
    dtype_out: np.dtype = np.int8,
    acc_max: int = 4, 
    gyro_max: int = 500,
    mag_max: Optional[int] = None) -> np.ndarray:

    data_copy = data.copy()
    data_copy[:, :, 0] = np.clip(data_copy[:, :, 0], -acc_max, acc_max)
    data_copy[:, :, 1] = np.clip(data_copy[:, :, 1], -acc_max, acc_max)
    data_copy[:, :, 2] = np.clip(data[:, :, 2], -acc_max, acc_max)

    data_copy[:, :, 3] = np.clip(data_copy[:, :, 3], -gyro_max, gyro_max)
    data_copy[:, :, 4] = np.clip(data_copy[:, :, 4], -gyro_max, gyro_max)
    data_copy[:, :, 5] = np.clip(data_copy[:, :, 5], -gyro_max, gyro_max)
    
    info = np.iinfo(dtype_out)
    min = info.min
    max = info.max

    data_copy[:, :, 0] = rescale(data_copy[:, :, 0], -acc_max, acc_max, min, max)
    data_copy[:, :, 1] = rescale(data_copy[:, :, 1], -acc_max, acc_max, min, max)
    data_copy[:, :, 2] = rescale(data_copy[:, :, 2], -acc_max, acc_max, min, max)

    data_copy[:, :, 3] = rescale(data_copy[:, :, 3], -gyro_max, gyro_max, min, max)
    data_copy[:, :, 4] = rescale(data_copy[:, :, 4], -gyro_max, gyro_max, min, max)
    data_copy[:, :, 5] = rescale(data_copy[:, :, 5], -gyro_max, gyro_max, min, max)

    data_copy = data_copy.astype(dtype_out)

    return data_copy
   