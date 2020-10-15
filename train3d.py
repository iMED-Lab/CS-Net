#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Lei Mou
# @File   : train3d.py
"""
Training script for CS-Net 3D
"""
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import datetime
import numpy as np

from model.csnet_3d import CSNet3D
from dataloader.MRABrainLoader import Data

from utils.train_metrics import metrics3d
from utils.losses import WeightedCrossEntropyLoss, DiceLoss
from utils.visualize import init_visdom_line, update_lines

args = {
    'root'      : '/home/user/name/Projects/',
    'data_path' : 'dataset/data dir(your own data path)/',
    'epochs'    : 200,
    'lr'        : 0.0001,
    'snapshot'  : 100,
    'test_step' : 1,
    'ckpt_path' : './checkpoint/',
    'batch_size': 2,
}

# # Visdom---------------------------------------------------------
# The initial values are defined by myself
X, Y = 0, 1.0  # for visdom
x_tp, y_tp = 0, 0
x_fn, y_fn = 0.4, 0.4
x_fp, y_fp = 0.4, 0.4
x_testtp, y_testtp = 0.0, 0.0
x_testdc, y_testdc = 0.0, 0.0
env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss", env="wce")
env1, panel1 = init_visdom_line(x_tp, y_tp, title="TPR", xlabel="iters", ylabel="TPR", env="wce")
env2, panel2 = init_visdom_line(x_fn, y_fn, title="FNR", xlabel="iters", ylabel="FNR", env="wce")
env3, panel3 = init_visdom_line(x_fp, y_fp, title="FPR", xlabel="iters", ylabel="FPR", env="wce")
env6, panel6 = init_visdom_line(x_testtp, y_testtp, title="DSC", xlabel="iters", ylabel="DSC", env="wce")
env4, panel4 = init_visdom_line(x_testtp, y_testtp, title="Test Loss", xlabel="iters", ylabel="Test Loss", env="wce")
env5, panel5 = init_visdom_line(x_testdc, y_testdc, title="Test TP", xlabel="iters", ylabel="Test TP", env="wce")
env7, panel7 = init_visdom_line(x_testdc, y_testdc, title="Test IoU", xlabel="iters", ylabel="Test IoU", env="wce")


def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    date = datetime.datetime.now().strftime("%Y-%m-%d-")
    torch.save(net, args['ckpt_path'] + 'CSNet3D_' + date + iter + '.pkl')
    print("{} Saved model to:{}".format("\u2714", args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    net = CSNet3D(classes=2, channels=1).cuda()
    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)

    # load train dataset
    train_data = Data(args['data_path'], train=True)
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=4, shuffle=True)

    critrion2 = WeightedCrossEntropyLoss().cuda()
    critrion = nn.CrossEntropyLoss().cuda()
    critrion3 = DiceLoss().cuda()
    # Start training
    print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))

    iters = 1
    for epoch in range(args['epochs']):
        net.train()
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()
            pred = net(image)
            loss_dice = critrion3(pred, label)
            label = label.squeeze(1)
            loss_ce = critrion(pred, label)
            loss_wce = critrion2(pred, label)
            loss = (loss_ce + 0.6 * loss_wce + 0.4 * loss_dice) / 3
            loss.backward()
            optimizer.step()
            tp, fn, fp, iou = metrics3d(pred, label, pred.shape[0])
            if (epoch % 2) == 0:
                print(
                    '\033[1;36m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tIoU:{6:.4f} '.format(
                        epoch + 1, iters, loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0],
                        iou / pred.shape[0]))
            else:
                print(
                    '\033[1;32m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tIoU:{6:.4f} '.format(
                        epoch + 1, iters, loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0],
                        iou / pred.shape[0]))

            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            X, x_tp, x_fn, x_fp, x_dc = iters, iters, iters, iters, iters
            Y, y_tp, y_fn, y_fp, y_dc = loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0], iou / \
                                        pred.shape[0]

            update_lines(env, panel, X, Y)
            update_lines(env1, panel1, x_tp, y_tp)
            update_lines(env2, panel2, x_fn, y_fn)
            update_lines(env3, panel3, x_fp, y_fp)
            update_lines(env6, panel6, x_dc, y_dc)

            # # --------------------------------------------------------------------------------------------

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)

        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))

        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_tp, test_fn, test_fp, test_dc = model_eval(net, critrion, iters)
            print("Average TP:{0:.4f}, average FN:{1:.4f},  average FP:{2:.4f}".format(test_tp, test_fn, test_fp))


def model_eval(net, critrion, iters):
    print("\033[1;30;43m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    TP, FN, FP, IoU = [], [], [], []
    file_num = 0
    with torch.no_grad():
        for idx, batch in enumerate(batchs_data):
            image = batch[0].float().cuda()
            label = batch[1].cuda()
            pred_val = net(image)
            label = label.squeeze(1)
            loss = critrion(pred_val, label)
            tp, fn, fp, iou = metrics3d(pred_val, label, pred_val.shape[0])
            print(
                "--- test TP:{0:.4f}    test FN:{1:.4f}    test FP:{2:.4f}    test IoU:{3:.4f}".format(tp, fn, fp, iou))
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            IoU.append(iou)
            file_num += 1
            # # start visdom images
            X, x_testtp, x_testdc = iters, iters, iters
            Y, y_testtp, y_testdc = loss.item(), tp / pred_val.shape[0], iou / pred_val.shape[0]
            update_lines(env4, panel4, X, Y)
            update_lines(env5, panel5, x_testtp, y_testtp)
            update_lines(env7, panel7, x_testdc, y_testdc)
            return np.mean(TP), np.mean(FN), np.mean(FP), np.mean(IoU)


if __name__ == '__main__':
    train()
