import numpy as np
import os
import torch.nn as nn
import torch
from PIL import ImageOps, Image
from sklearn.metrics import confusion_matrix


# using manual threshold during training, but OTSU threshold in off line test
def threshold(image):
    image[image >= 100] = 255
    image[image < 100] = 0
    return image


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 255) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 255)))
    TP = np.float(np.sum((pred == 255) & (gt == 255)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def metrics(pred, label, batch_size):
    # pred = torch.argmax(pred, dim=1) # for CE Loss series
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    Acc, SEn = 0., 0.
    for i in range(batch_size):
        img = outputs[i, :, :]
        gt = labels[i, :, :]
        acc, sen = get_acc(img, gt)
        Acc += acc
        SEn += sen
    return Acc, SEn



def get_acc(image, label):
    image = threshold(image)
    FP, FN, TP, TN = numeric_score(image, label)
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)
    sen = (TP) / (TP + FN + 1e-10)
    return acc, sen
