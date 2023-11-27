import glob
from PIL import Image
import torch.nn as nn
import torch
import copy
import argparse
import numpy as np
import os
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
from albumentations import Resize
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms as abtransforms
import torch.backends.cudnn as cudnn
import yaml
import ast
from models.resmtunet import ResMTUnet
import pydensecrf.densecrf as dcrf
Image.MAX_IMAGE_PIXELS = 933120000


cudnn.benchmark = True

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--device', type=str, help='device:ID',
                    default='cuda:2')  # for gpu use the string format of 'cuda:#', e.g: cuda:0
parser.add_argument('--n_classes', type=int, help='n_classes', default=3)
parser.add_argument('--num_epoch', type=int, help='num_epoch', default=30)
parser.add_argument('--img_size', type=int, help='img_size', default=512)
parser.add_argument('--data_path', type=str, help='data_path', default='/path/to/data_path')
parser.add_argument('--save_dir', type=str, help='save_dir', default='infer_exp/ms_crf')
parser.add_argument('--arch', type=str, help='model architecture', default='Unet')
parser.add_argument('--encoder', type=str, help='model encoder', default='mit_b5')
parser.add_argument('--ms', action='store_true', help='multi scale')
parser.add_argument('--crf', action='store_true', help='condicion random field')
parser.add_argument('--ckpt_path', type=str, help='ckpt path',
                    default="exp0105/mode_0/2023-01-05 19:14:21.570560/best_model_wts.pth")


parser.add_argument('--gt_prob', type=lambda x: ast.literal_eval(x), nargs='+', help='gt_prob', default=[0.5, 0.5, 0.5, 0.5])
parser.add_argument('--Gau_sxy', type=int, help='Gaussian sxy', default=20)
parser.add_argument('--Bi_sxy', type=int, help='Bilateral sxy', default=15)
parser.add_argument('--Bi_srgb', type=int, help='Bilateral srgb', default=25)
parser.add_argument('--Gau_cp', type=int, help='Gaussian cp', default=3)
parser.add_argument('--Bi_cp', type=int, help='Bilateral cp', default=10)


def convert_to_color(output_, img):
    index_map = np.argmax(output_, axis=0)
    infer_result = np.zeros_like(img).astype(np.uint8)
    infer_result[index_map == 0] = 255
    infer_result[index_map == 1] = np.array([0, 255, 0])
    infer_result[index_map == 2] = np.array([0, 0, 255])
    infer_result[index_map == 3] = np.array([255, 0, 0])
    
    return infer_result


def process_annotation(images):
    merged_colors = []
    all_labels = []

    for anno_rgb in images:
        anno_rgb = anno_rgb.astype(np.uint32)
        anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

        colors, labels = np.unique(anno_lbl, return_inverse=True)

        merged_colors.extend(colors)
        all_labels.append(anno_lbl)

    merged_colors = np.unique(merged_colors)

    colorize = np.empty((len(merged_colors), 3), np.uint8)
    colorize[:, 0] = (merged_colors & 0x0000FF)
    colorize[:, 1] = (merged_colors & 0x00FF00) >> 8
    colorize[:, 2] = (merged_colors & 0xFF0000) >> 16

    n_labels = len(merged_colors)

    transformed_labels = []
    for labels in all_labels:
        transformed_labels.append(np.searchsorted(merged_colors, labels))

    return colorize, transformed_labels, n_labels

def unary_from_labels(labels, n_labels, gt_prob, zero_unsure=True):
    """
    Simple classifier that is 50% certain that the annotation is correct.
    (same as in the inference example).


    Parameters
    ----------
    labels: numpy.array
        The label-map, i.e. an array of your data's shape where each unique
        value corresponds to a label.
    n_labels: int
        The total number of labels there are.
        If `zero_unsure` is True (the default), this number should not include
        `0` in counting the labels, since `0` is not a label!
    gt_prob: float
        The certainty of the ground-truth (must be within (0,1)).
    zero_unsure: bool
        If `True`, treat the label value `0` as meaning "could be anything",
        i.e. entries with this value will get uniform unary probability.
        If `False`, do not treat the value `0` specially, but just as any
        other class.
    """
    assert 0 < gt_prob < 1, "`gt_prob must be in (0,1)."

    labels = labels.flatten()

    n_energy = -np.log((1.0 - gt_prob) / (n_labels - 1))
    p_energy = -np.log(gt_prob)

    # Note that the order of the following operations is important.
    # That's because the later ones overwrite part of the former ones, and only
    # after all of them is `U` correct!
    U = np.full((n_labels, len(labels)), n_energy, dtype='float32')
    U[labels - 1 if zero_unsure else labels, np.arange(U.shape[1])] = p_energy

    # Overwrite 0-labels using uniform probability, i.e. "unsure".
    if zero_unsure:
        U[:, labels == 0] = -np.log(1.0 / n_labels)

    return U

def CRFs(img, anno_rgb, anno_rgb_ms, args):
    if len(args.gt_prob) == 4:
        gt_prob = args.gt_prob
    elif len(args.gt_prob) == 1 and len(args.gt_prob[0]) == 4:
        gt_prob = args.gt_prob[0]
    else:
        raise TypeError
    Gau_sxy = args.Gau_sxy
    Bi_sxy = args.Bi_sxy
    Bi_srgb = args.Bi_srgb
    Gau_cp = args.Gau_cp
    Bi_cp = args.Bi_cp
    
    # colorize, labels, n_labels = process_annotation(anno_rgb)
    colorize, labels, n_labels = process_annotation([anno_rgb] + [convert_to_color(item, img) for item in anno_rgb_ms])
        
    # Setup the CRF model.
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # Get unary potentials (neg log probability).
    U = unary_from_labels(labels[0], n_labels, gt_prob=gt_prob[0], zero_unsure=None)
    ################################
    for k, lbls in enumerate(labels[1:]):
        U += unary_from_labels(lbls, n_labels, gt_prob=gt_prob[k+1], zero_unsure=None)
    ################################
    d.setUnaryEnergy(U)
    
    # Add a term that penalizes isolated small segments spatially
    # -- enforces more spatially consistent segmentation.
    d.addPairwiseGaussian(sxy=(Gau_sxy, Gau_sxy), compat=Gau_cp, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Add a color-independent term, purely based on position -- features are (x,y,r,g,b)
    d.addPairwiseBilateral(sxy=(Bi_sxy, Bi_sxy), srgb=(Bi_srgb, Bi_srgb, Bi_srgb),
                           rgbim=np.ascontiguousarray(img), compat=Bi_cp,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the predicted image back to the corresponding colors and save the image.
    MAP = colorize[MAP, :]

    return MAP.reshape(img.shape)


def main(args):
    """
    :param args:
    :return:
    """
    os.makedirs(args.save_dir, exist_ok=True)
    # 将参数转换为字典格式  
    args_dict = vars(args)  
    # 打开'config.yaml'文件，将参数写入  
    with open(args.save_dir + '_config.yaml', 'w') as file:  
        yaml.dump(args_dict, file)  

    val_transform = Compose([
        Resize(args.img_size, args.img_size),
        abtransforms.Normalize(),
    ])

    '''2. 构建模型'''
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        # activation='sigmoid',  # activation function, default is None
        classes=args.n_classes,  # define number of output labels
    )

    model_ft = ResMTUnet(encoder_name='mit_b5',
                         resnet_backbone_name='resnet34',
                         classes=4,
                         aux_params=aux_params)  # MAnet, FPN, PSPNet

    device = torch.device(args.device)
    model_ft = model_ft.to(device)

    best_model_wts_ = torch.load(args.ckpt_path, map_location='cpu')
    best_model_wts = copy.deepcopy(best_model_wts_)
    for name in best_model_wts_:
        if 'fusion_conv' in name:
            new_name = name.replace('fusion_conv', 'fusion_conv.0')
            best_model_wts[new_name] = best_model_wts[name]
            del best_model_wts[name]
    del best_model_wts_

    '''5. start testing'''
    model_ft.load_state_dict(best_model_wts)
    device = torch.device(args.device)
    model_ft = model_ft.to(device)
    model_ft.eval()

    img_paths = glob.glob(os.path.join(args.data_path, '*.jpg'))
    if args.ms:
        CROP_SIZE_list = [512 * 2, 512 * 3, 512 * 6]
    else:
        CROP_SIZE_list = [512 * 3]
    for img_path in tqdm(img_paths):
        img = Image.open(img_path).convert('RGB')
        img = Image.fromarray(np.array(img)[:512*6, :512*6, :])
        W, H = img.size
        output = []
        for CROP_SIZE in CROP_SIZE_list:

            ''' --------- 切成小图输入 --------- '''
            kw = int(np.ceil((W - CROP_SIZE) / CROP_SIZE) + 1)
            kh = int(np.ceil((H - CROP_SIZE) / CROP_SIZE) + 1)
            output_tensor = np.zeros((args.n_classes+1, kh * args.img_size, kw * args.img_size))
            output_tensor = np.zeros((args.n_classes+1, kh * args.img_size, kw * args.img_size))
            for w in range(kw):
                for h in range(kh):
                    cropped_img = img.crop((w * CROP_SIZE, h * CROP_SIZE,
                                            (w + 1) * CROP_SIZE,
                                            (h + 1) * CROP_SIZE))  # (left, upper, right, lower)
                    img_np = val_transform(image=np.array(cropped_img))['image']
                    img_tensor = transforms.ToTensor()(img_np).to(device).unsqueeze(0)
                    with torch.no_grad():
                        mask, label = model_ft(img_tensor)
                        heatmap = nn.Softmax(dim=0)(mask[0, ...]).detach().cpu().numpy()
                    output_tensor[:, h * args.img_size:(h + 1) * args.img_size, \
                    w * args.img_size:(w + 1) * args.img_size] = heatmap
            output1 = cv2.resize(output_tensor.transpose(1, 2, 0), None, fx=CROP_SIZE / args.img_size, fy=CROP_SIZE / args.img_size)
            output.append(output1.transpose(2, 0, 1)[:, :H, :W])
        output_ = np.mean(output, axis=0)
        infer_result = convert_to_color(output_, img)

        if args.crf:
            infer_result = CRFs(np.array(img), infer_result, output, args)
        Image.fromarray(infer_result).save(os.path.join(args.save_dir, 
                                                        os.path.basename(img_path).replace('.jpg', '.png')))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
