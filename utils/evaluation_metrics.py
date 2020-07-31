"""
Evaluation metrics
"""

import numpy as np
import sklearn.metrics as metrics
import os
import glob
import cv2
from PIL import Image


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 1) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def numeric_score_fov(pred, gt, mask):
    FP = np.float(np.sum((pred == 1) & (gt == 0) & (mask == 1)))
    FN = np.float(np.sum((pred == 0) & (gt == 1) & (mask == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1) & (mask == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0) & (mask == 1)))
    return FP, FN, TP, TN


def AUC(path):
    all_auc = 0.
    file_num = 0
    for file in glob.glob(os.path.join(path, 'pred', '*pred.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-9] + '.png'
        label_path = os.path.join(path, 'label', label_name)

        mask_path = '/path/to/FOV/mask/'

        pred_image = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=-1)
        mask = cv2.imread(mask_path, flags=-1)

        # with FOV
        label_fov = []
        pred_fov = []
        w, h = pred_image.shape
        for i in range(w):
            for j in range(h):
                if mask[i, j] == 255:
                    label_fov.append(label[i, j])
                    pred_fov.append(pred_image[i, j])
        pred_image = (np.asarray(pred_fov)) / 255
        label = np.uint8((np.asarray(label_fov)) / 255)

        # pred_image = pred_image.flatten() / 255
        # label = np.uint8(label.flatten() / 255)

        auc_score = metrics.roc_auc_score(label, pred_image)
        all_auc += auc_score
        file_num += 1
    avg_auc = all_auc / file_num
    return avg_auc


def DSC(path):
    all_dsc = 0.
    file_num = 0
    for file in glob.glob(os.path.join(path, 'pred', '*otsu.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-14] + '.png'
        label_path = os.path.join(path, 'label', label_name)

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=-1)

        pred = pred // 255
        label = label // 255

        FP, FN, TP, TN = numeric_score(pred, label)
        dsc = 2 * TP / (FP + 2 * TP + FN)
        all_dsc += dsc
        file_num += 1
    avg_dsc = all_dsc / file_num
    return avg_dsc


def AccSenSpe(path):
    all_sen = []
    all_acc = []
    all_spe = []
    for file in glob.glob(os.path.join(path, 'pred', '*otsu.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-14] + '.png'
        label_path = os.path.join(path, 'label', label_name)

        mask_path = '/path/to/FOV/mask/'

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=-1)
        mask = cv2.imread(mask_path, flags=-1)

        pred = pred // 255
        label = label // 255
        mask = mask // 255

        FP, FN, TP, TN = numeric_score(pred, label)
        acc = (TP + TN) / (TP + FP + TN + FN)
        sen = TP / (TP + FN)
        spe = TN / (TN + FP)
        all_acc.append(acc)
        all_sen.append(sen)
        all_spe.append(spe)
    avg_acc, avg_sen, avg_spe = np.mean(all_acc), np.mean(all_sen), np.mean(all_spe)
    var_acc, var_sen, var_spe = np.var(all_acc), np.var(all_sen), np.var(all_spe)
    return avg_acc, var_acc, avg_sen, var_sen, avg_spe, var_spe


def FDR(path):
    all_fdr = []
    for file in glob.glob(os.path.join(path, 'pred', '*otsu.png')):
        base_name = os.path.basename(file)
        label_name = base_name[:-14] + '.png'
        label_path = os.path.join(path, 'label', label_name)

        pred = cv2.imread(file, flags=-1)
        label = cv2.imread(label_path, flags=-1)

        pred = pred // 255
        label = label // 255

        FP, FN, TP, TN = numeric_score(pred, label)
        fdr = FP / (FP + TP)
        all_fdr.append(fdr)
    return np.mean(all_fdr), np.var(all_fdr)


if __name__ == '__main__':
    # predicted root path
    path = './assets/Padova1/'
    # auc = AUC(path)
    acc, var_acc, sen, var_sen, spe, var_spe = AccSenSpe(path)
    fdr, var_fdr = FDR(path)
    print("sen:{0:.4f} +- {1:.4f}".format(sen, var_sen))
    print("fdr:{0:.4f} +- {1:.4f}".format(fdr, var_fdr))
    # print("acc:{0:.4f}".format(acc))
    # print("sen:{0:.4f}".format(sen))
    # print("spe:{0:.4f}".format(spe))
    # print("auc:{0:.4f}".format(auc))
