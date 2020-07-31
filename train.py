"""
Training script for CS-Net
"""
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import visdom
import numpy as np
from model.csnet import CSNet
from dataloader.drive import Data
from utils.train_metrics import metrics
from utils.visualize import init_visdom_line, update_lines

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

args = {
    'root'      : '',
    'data_path' : 'dataset/DRIVE/',
    'epochs'    : 1000,
    'lr'        : 0.0001,
    'snapshot'  : 100,
    'test_step' : 1,
    'ckpt_path' : 'checkpoint/',
    'batch_size': 8,
}

# # Visdom---------------------------------------------------------
X, Y = 0, 0.5  # for visdom
x_acc, y_acc = 0, 0
x_sen, y_sen = 0, 0
env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss")
env1, panel1 = init_visdom_line(x_acc, y_acc, title="Accuracy", xlabel="iters", ylabel="accuracy")
env2, panel2 = init_visdom_line(x_sen, y_sen, title="Sensitivity", xlabel="iters", ylabel="sensitivity")
# # ---------------------------------------------------------------

def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, args['ckpt_path'] + 'CS_Net_DRIVE_' + str(iter) + '.pkl')
    print('--->saved model:{}<--- '.format(args['root'] + args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # set the channels to 3 when the format is RGB, otherwise 1.
    net = CSNet(classes=1, channels=3).cuda()
    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)
    critrion = nn.MSELoss().cuda()
    # critrion = nn.CrossEntropyLoss().cuda()
    print("---------------start training------------------")
    # load train dataset
    train_data = Data(args['data_path'], train=True)
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=2, shuffle=True)

    iters = 1
    accuracy = 0.
    sensitivty = 0.
    for epoch in range(args['epochs']):
        net.train()
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()
            pred = net(image)
            # pred = pred.squeeze_(1)
            loss = critrion(pred, label)
            loss.backward()
            optimizer.step()
            acc, sen = metrics(pred, label, pred.shape[0])
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}'.format(epoch + 1,
                                                                                     iters, loss.item(),
                                                                                     acc / pred.shape[0],
                                                                                     sen / pred.shape[0]))
            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            X, x_acc, x_sen = iters, iters, iters
            Y, y_acc, y_sen = loss.item(), acc / pred.shape[0], sen / pred.shape[0]
            update_lines(env, panel, X, Y)
            update_lines(env1, panel1, x_acc, y_acc)
            update_lines(env2, panel2, x_sen, y_sen)
            # # --------------------------------------------------------------------------------------------

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, epoch + 1)

        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_acc, test_sen = model_eval(net)
            print("Average acc:{0:.4f}, average sen:{1:.4f}".format(test_acc, test_sen))

            if (accuracy > test_acc) & (sensitivty > test_sen):
                save_ckpt(net, epoch + 1 + 8888888)
                accuracy = test_acc
                sensitivty = test_sen


def model_eval(net):
    print("Start testing model...")
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    Acc, Sen = [], []
    file_num = 0
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].float().cuda()
        pred_val = net(image)
        acc, sen = metrics(pred_val, label, pred_val.shape[0])
        print("\t---\t test acc:{0:.4f}    test sen:{1:.4f}".format(acc, sen))
        Acc.append(acc)
        Sen.append(sen)
        file_num += 1
        # for better view, add testing visdom here.
        return np.mean(Acc), np.mean(Sen)


if __name__ == '__main__':
    train()
