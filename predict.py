import torch
from torchvision import transforms
from PIL import Image, ImageOps

import numpy as np
import scipy.misc as misc
import os
import glob

from utils.misc import thresh_OTSU, ReScaleSize, Crop
from utils.model_eval import eval

DATABASE = './DRIVE/'
#
args = {
    'root'     : './dataset/' + DATABASE,
    'test_path': './dataset/' + DATABASE + 'test/',
    'pred_path': 'assets/' + 'DRIVE/',
    'img_size' : 512
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def rescale(img):
    w, h = img.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    img = img.crop(box)
    return img


def ReScaleSize_DRIVE(image, re_size=512):
    w, h = image.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    image = image.crop(box)
    image = image.resize((re_size, re_size))
    return image  # , origin_w, origin_h


def ReScaleSize_STARE(image, re_size=512):
    w, h = image.size
    max_len = max(w, h)
    new_w, new_h = max_len, max_len
    delta_w = new_w - w
    delta_h = new_h - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding, fill=0)
    # origin_w, origin_h = w, h
    image = image.resize((re_size, re_size))
    return image  # , origin_w, origin_h


def load_nerve():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'orig', '*.tif')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'orig', basename)
        label_name = os.path.join(args['test_path'], 'mask2', file_name + '_centerline_overlay.tif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_drive():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.tif')):
        basename = os.path.basename(file)
        file_name = basename[:3]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], '1st_manual', file_name + 'manual1.gif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_stare():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.ppm')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'labels-ah', file_name + '.ah.ppm')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_padova1():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.tif')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'label2', file_name + '_centerline_overlay.tif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_octa():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.png')):
        basename = os.path.basename(file)
        file_name = basename[:-4]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'label', file_name + '_nerve_ann.tif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_net():
    net = torch.load('./checkpoint/xxxx.pkl')
    return net


def save_prediction(pred, filename=''):
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    mask = pred.data.cpu().numpy() * 255
    mask = np.transpose(np.squeeze(mask, axis=0), [1, 2, 0])
    mask = np.squeeze(mask, axis=-1)
    misc.imsave(save_path + filename + '.png', mask)


def predict():
    net = load_net()
    # images, labels = load_nerve()
    images, labels = load_drive()
    # images, labels = load_stare()
    # images, labels = load_padova1()
    # images, labels = load_octa()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    with torch.no_grad():
        net.eval()
        for i in range(len(images)):
            print(images[i])
            name_list = images[i].split('/')
            index = name_list[-1][:-4]
            image = Image.open(images[i])
            # image=image.convert("RGB")
            label = Image.open(labels[i])
            image, label = center_crop(image, label)

            # for other retinal vessel
            # image = rescale(image)
            # label = rescale(label)
            # image = ReScaleSize_STARE(image, re_size=args['img_size'])
            # label = ReScaleSize_DRIVE(label, re_size=args['img_size'])

            # for OCTA
            # image = Crop(image)
            # image = ReScaleSize(image)
            # label = Crop(label)
            # label = ReScaleSize(label)

            # label = label.resize((args['img_size'], args['img_size']))
            # if cuda
            image = transform(image).cuda()
            # image = transform(image)
            image = image.unsqueeze(0)
            output = net(image)

            save_prediction(output, filename=index + '_pred')
            print("output saving successfully")


if __name__ == '__main__':
    predict()
    thresh_OTSU(args['pred_path'] + 'pred/')
