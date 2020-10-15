import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from tqdm import tqdm
import SimpleITK as sitk
from utils.misc import get_spacing

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

DATABASE = 'VascuSynth3/'
#
args = {
    'root'     : './dataset/' + DATABASE,
    'test_path': './dataset/' + DATABASE + 'test/',
    'pred_path': 'assets/' + 'VascuSynth3/',
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


def load_3d():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.mha')):
        basename = os.path.basename(file)
        file_name = basename[:-8]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'label', file_name + 'gt.mha')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_net():
    net = torch.load('/home/imed/Research/Attention/checkpoint/model.pkl')
    return net


def save_prediction(pred, filename='', spacing=None):
    pred = torch.argmax(pred, dim=1)
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    # for MSELoss()
    mask = (pred.data.cpu().numpy() * 255).astype(np.uint8)

    # thresholding
    # mask[mask >= 100] = 255
    # mask[mask < 100] = 0

    # mask = (mask.squeeze(0)).squeeze(0)  # 3D numpy array
    mask = mask.squeeze(0)  # for CE Loss
    # image = nib.Nifti1Image(np.int32(mask), affine)
    # nib.save(image, save_path + filename + ".nii.gz")
    mask = sitk.GetImageFromArray(mask)
    # if spacing is not None:
    #     mask.SetSpacing(spacing)
    sitk.WriteImage(mask, os.path.join(save_path + filename + ".mha"))


def save_probability(pred, label, filename=""):
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    # # for MSELoss()
    # mask = (pred.data.cpu().numpy() * 255)  # .astype(np.uint8)
    #
    # mask = mask.squeeze(0)
    # class0 = mask[0, :, :, :]
    # class1 = mask[1, :, :, :]
    # label = label / 255
    # class0 = class0 * label
    # class1 = class1 * label
    #
    # probability = class0 + class1

    probability = F.softmax(pred, dim=1)
    probability.squeeze_(0)
    class0 = probability[0, :, :, :]
    class1 = probability[1, :, :, :]
    class0 = sitk.GetImageFromArray(class0)
    class1 = sitk.GetImageFromArray(class1)
    sitk.WriteImage(class1, os.path.join(save_path + filename + "class1.mha"))


def save_label(label, index, spacing=None):
    label_path = args['pred_path'] + 'label/'
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    label = sitk.GetImageFromArray(label)
    if spacing is not None:
        label.SetSpacing(spacing)
    sitk.WriteImage(label, os.path.join(label_path, index + ".mha"))


def predict():
    net = load_net()
    images, labels = load_3d()
    with torch.no_grad():
        net.eval()
        for i in tqdm(range(len(images))):
            name_list = images[i].split('/')
            index = name_list[-1][:-4]
            image = sitk.ReadImage(images[i])
            image = sitk.GetArrayFromImage(image).astype(np.float32)
            image = image / 255
            label = sitk.ReadImage(labels[i])
            label = sitk.GetArrayFromImage(label).astype(np.int64)
            # label = label / 255
            # VascuSynth
            # image = image[2:98, 2:98, 2:98]
            # label = label[2:98, 2:98, 2:98]
            save_label(label, index)
            # if cuda
            image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0)
            image = image.cuda()
            output = net(image)
            save_prediction(output, filename=index + '_pred', spacing=None)


if __name__ == '__main__':
    predict()
