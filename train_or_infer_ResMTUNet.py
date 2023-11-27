from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import argparse
import numpy as np
import os
import datetime
from tqdm import tqdm
import yaml
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations import RandomRotate90, Resize
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms as abtransforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from pprint import pprint
from models.resmtunet import ResMTUnet

cudnn.benchmark = True

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--device', type=str, help='device:ID',
                    default='cuda:0')  # for gpu use the string format of 'cuda:#', e.g: cuda:0
parser.add_argument('--n_classes', type=int, help='n_classes', default=3)
parser.add_argument('--num_epoch', type=int, help='num_epoch', default=30)
parser.add_argument('--img_size', type=int, help='img_size', default=512)
parser.add_argument('--batch_size', type=int, help='batch size', default=7)
parser.add_argument('--data_path', type=str, help='data_path', default='/path/to/BRACS')
parser.add_argument('--mask_path', type=str, help='mask_path', default='/path/to/mask')
parser.add_argument('--att_mask_mode', type=int, default=0,
                    help='if mode == 6, then 6 --> 110. train: 1, val: 1, test: 0')
parser.add_argument('--exp_name', type=str, help='exp_name', default='experiment/exp1')
parser.add_argument('--arch', type=str, help='model architecture', default='Unet')
parser.add_argument('--encoder', type=str, help='model encoder', default='mit_b5')

parser.add_argument('--mode', type=str, help='train or val', default='train')
parser.add_argument('--ckpt_path', type=str, help='ckpt path', default="/path/to/ckpt.pth")

parser.add_argument('--cls_loss_w', type=float, help='loss weight for classification', default=0.5)
parser.add_argument('--seg_loss_w', type=float, help='loss weight for segmentation', default=1.0)


def iou_score(output, target, num_classes):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.argmax(output, dim=1).cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    iou_scores = []
    dice_scores = []
    for c in range(1, num_classes+1):
        intersection = np.sum((output == c) & (target == c))
        union = np.sum((output == c) | (target == c))
        if np.sum(target == c) == 0:
            # iou_scores.append(0.0)
            continue
        else:
            iou = (intersection + smooth) / (union + smooth)
            dice = (2 * iou) / (iou + 1)
            iou_scores.append(iou)
            dice_scores.append(dice)
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    return mean_iou, mean_dice


class MTDataset(Dataset):
    def __init__(self, img_ids, phase, img_dir, mask_dir, num_classes, img_size=512, transform=None, att_mask=True):
        self.img_ids = img_ids
        self.phase = phase
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform
        self.att_mask = att_mask
        self.img_size = img_size
        self.class_to_idx = {
            '0_N': 0,
            '1_PB': 0,
            '2_UDH': 1,
            '3_FEA': 1,
            '4_ADH': 1,
            '5_DCIS': 1,
            '6_IC': 2}
        self.class_list = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def overlay_img_with_att_mask(img, att_mask_dir, w):
        normal_att_mask_path = os.path.join(att_mask_dir, 'normal_att_mask.npy')
        normal_att_mask = np.load(normal_att_mask_path)
        tumor_att_mask_path = os.path.join(att_mask_dir, 'tumor_att_mask.npy')
        tumor_att_mask = np.load(tumor_att_mask_path)

        # ----------------- 红绿掩膜 -----------------
        normal_att_mask = (normal_att_mask * 255).astype(np.uint8)
        normal_att_mask = normal_att_mask[:, :, None] * np.array([[[0, 1, 0]]])
        tumor_att_mask = (tumor_att_mask * 255).astype(np.uint8)
        tumor_att_mask = tumor_att_mask[:, :, None] * np.array([[[1, 0, 0]]])
        img = w * (normal_att_mask + tumor_att_mask) + (1 - w) * np.array(img)
        img = img.astype(np.uint8)

        return img

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        cls_name = '%d_%s' % (self.class_list.index(img_id.split('_')[-2]), img_id.split('_')[-2])
        img = Image.open(os.path.join(self.img_dir, cls_name, img_id)).convert('RGB')
        img = np.array(img.resize((self.img_size, self.img_size)))

        masks_dir = os.path.join(self.mask_dir, cls_name, img_id[:-len('.png')])

        if self.att_mask:
            if self.phase != 'train':
                img = self.overlay_img_with_att_mask(img, masks_dir, w=0.2)
            else:
                img = self.overlay_img_with_att_mask(img, masks_dir, w=np.random.random()*0.4)

        mask_path = os.path.join(masks_dir, 'mask.npy')
        _mask = (np.load(mask_path) > 0.5).astype(np.float32)
        mask = _mask * (self.class_to_idx[cls_name] + 1)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.transpose(2, 0, 1)
        mask = torch.tensor(mask).long()

        return img, mask, self.class_to_idx[cls_name]


def train_one_epoch(train_loader, model_ft, optimizer, criterion, device, loss_w):
    model_ft.train()
    train_loss, train_acc, train_iou, train_dice = 0.0, 0.0, 0.0, 0.0
    seg_criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

    with tqdm(train_loader, desc=' Training') as tbar:
        for imgs, masks, labels in tbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            mask_outputs, label_outputs = model_ft(imgs)
            loss_seg = seg_criterion(mask_outputs, masks)
            loss_cls = criterion(label_outputs, labels)
            loss = loss_w[1] * loss_seg + loss_w[0] * loss_cls

            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            train_loss += step_loss * imgs.size(0)

            _, preds = torch.max(label_outputs, 1)
            step_acc_sum = torch.sum(preds == labels.detach())
            train_acc += step_acc_sum
            # one_hot_mask = F.one_hot(masks, num_classes=4).permute(0, 3, 1, 2)
            iou, dice = iou_score(mask_outputs, masks, args.n_classes)
            train_iou += iou * imgs.size(0)
            train_dice += dice * imgs.size(0)

            tbar.set_postfix(iou=iou,
                             dice=dice,
                             loss=step_loss,
                             loss_cls=loss_cls.item(),
                             acc=step_acc_sum.item() / imgs.size(0))
            tbar.update()
        torch.cuda.empty_cache()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    train_iou /= len(train_loader.dataset)
    train_dice /= len(train_loader.dataset)

    return train_iou, train_dice, train_acc


def val_one_epoch(val_loader, model_ft, criterion, device, loss_w):
    model_ft.eval()
    val_loss, val_acc, val_iou, val_dice = 0.0, 0.0, 0.0, 0.0
    seg_criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    with tqdm(val_loader, desc=' Val') as tbar:
        for imgs, masks, labels in tbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                mask_outputs, label_outputs = model_ft(imgs)
            loss_seg = seg_criterion(mask_outputs, masks)
            loss_cls = criterion(label_outputs, labels)
            loss = loss_w[1] * loss_seg + loss_w[0] * loss_cls

            step_loss = loss.item()
            val_loss += step_loss * imgs.size(0)

            _, preds = torch.max(label_outputs, 1)
            step_acc_sum = torch.sum(preds == labels.detach())
            val_acc += step_acc_sum
            # one_hot_mask = F.one_hot(masks, num_classes=4).permute(0, 3, 1, 2)
            iou, dice = iou_score(mask_outputs, masks, args.n_classes)
            val_iou += iou * imgs.size(0)
            val_dice += dice * imgs.size(0)

            tbar.set_postfix(iou=iou,
                             dice=dice,
                             loss=step_loss,
                             loss_cls=loss_cls.item(),
                             acc=step_acc_sum.item() / imgs.size(0))
            tbar.update()
        torch.cuda.empty_cache()

    val_iou /= len(val_loader.dataset)
    val_dice /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    return val_iou, val_dice, val_acc

def test_model(model_ft, test_loader, save_dir, args):

    device = torch.device(args.device)
    model_ft = model_ft.to(device)
    model_ft.eval()

    all_test_logits = []
    all_test_labels = []
    test_iou, test_iou_ic, count_ic = 0.0, 0.0, 0
    test_dice, test_dice_ic= 0.0, 0.0
    for imgs, masks, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            mask_outputs, logits = model_ft(imgs)
        all_test_logits.append(logits)
        all_test_labels.append(labels)
        # one_hot_mask = F.one_hot(masks, num_classes=4).permute(0, 3, 1, 2)
        iou, dice = iou_score(mask_outputs, masks, args.n_classes)
        test_iou += iou * imgs.size(0)
        test_dice += dice * imgs.size(0)
        if torch.sum(labels == 2).item() > 0:
            iou_ic, dice_ic = iou_score(mask_outputs[labels == 2], masks[labels == 2], args.n_classes)
            test_iou_ic += iou_ic * imgs[labels == 2].size(0)
            test_dice_ic += dice_ic * imgs[labels == 2].size(0)
            count_ic += imgs[labels == 2].size(0)

    test_iou /= len(test_loader.dataset)
    test_iou_ic /= count_ic
    test_dice /= len(test_loader.dataset)
    test_dice_ic /= count_ic

    all_test_logits = torch.cat(all_test_logits).cpu()
    all_test_preds = torch.argmax(all_test_logits, dim=1).numpy()
    all_test_labels = torch.cat(all_test_labels).cpu().numpy()

    accuracy = accuracy_score(all_test_labels, all_test_preds)
    weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
    report = classification_report(all_test_labels, all_test_preds, digits=4)
            
    error_img_idx = []
    for k, (th, pr) in enumerate(zip(all_test_labels, all_test_preds)):
        if th != pr:
            error_img_idx.append((test_loader.dataset.img_ids[k], (th, pr)))

    print('Test weighted F1 score {}'.format(weighted_f1_score))
    print('Test accuracy {}'.format(accuracy))
    print('Test classification report {}'.format(report))
    print('test_iou = %.4f, test_iou_ic = %.4f' % (test_iou, test_iou_ic))
    print('test_dice = %.4f, test_dice_ic = %.4f' % (test_dice, test_dice_ic))
    
    test_report_save_path = os.path.join(save_dir, 'test_report.txt')
    with open(test_report_save_path, 'w', encoding="utf-8") as f:
        print('Test weighted F1 score {}'.format(weighted_f1_score), file=f)
        print('Test accuracy {}'.format(accuracy), file=f)
        print('Test classification report {}'.format(report), file=f)
        print('test_iou = %.4f, test_iou_ic = %.4f' % (test_iou, test_iou_ic), file=f)
        print('test_dice = %.4f, test_dice_ic = %.4f' % (test_dice, test_dice_ic), file=f)
        print(error_img_idx, file=f)


def main(args):
    """
    :param args:
    :return:
    """
    '''1. 创建数据加载器'''
    save_dir = os.path.join(args.exp_name, str(datetime.datetime.now()))
    os.makedirs(save_dir, exist_ok=True)
    args_dict = vars(args)  
    with open(save_dir + '.yaml', 'w') as file:  
        yaml.dump(args_dict, file)  
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
        ], p=0.2),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        Resize(args.img_size, args.img_size),
        abtransforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(args.img_size, args.img_size),
        abtransforms.Normalize(),
    ])

    train_img_ids = glob.glob(os.path.join(args.data_path, 'train', '*', '*.png'))
    train_img_ids = [p.split('/')[-1] for p in train_img_ids]

    val_img_ids = glob.glob(os.path.join(args.data_path, 'val', '*', '*.png'))
    val_img_ids = [p.split('/')[-1] for p in val_img_ids]

    test_img_ids = glob.glob(os.path.join(args.data_path, 'test', '*', '*.png'))
    test_img_ids = [p.split('/')[-1] for p in test_img_ids]

    train_dataset = MTDataset(
        img_ids=train_img_ids,
        phase='train',
        img_dir=os.path.join(args.data_path, 'train'),
        mask_dir=os.path.join(args.mask_path, 'train'),
        num_classes=args.n_classes,
        img_size=args.img_size,
        transform=train_transform,
        att_mask=args.att_mask_mode // 4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False)

    val_dataset = MTDataset(
        img_ids=val_img_ids,
        phase='val',
        img_dir=os.path.join(args.data_path, 'val'),
        mask_dir=os.path.join(args.mask_path, 'val'),
        num_classes=args.n_classes,
        img_size=args.img_size,
        transform=val_transform,
        att_mask=(args.att_mask_mode % 4) // 2)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8 * args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False)

    test_dataset = MTDataset(
        img_ids=test_img_ids,
        phase='test',
        img_dir=os.path.join(args.data_path, 'test'),
        mask_dir=os.path.join(args.mask_path, 'test'),
        num_classes=args.n_classes,
        img_size=args.img_size,
        transform=val_transform,
        att_mask=(args.att_mask_mode % 4) % 2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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
    loss_w = [args.cls_loss_w, args.seg_loss_w]

    if args.mode == 'train':
        '''3. 损失优化学习策略'''
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

        best_model_wts = copy.deepcopy(model_ft.state_dict())
        best_index = {'acc': 0.0, 'iou': 0.0}

        '''4. 训练循环'''

        
        for epoch in range(args.num_epoch):
            # 训练，验证，输出，保存
            # 训练：数据，模型，优化器，损失函数，设备；损失、性能
            print('---- Epoch: %d / %d ----' % (epoch, args.num_epoch - 1))
            train_iou, train_dice, train_acc = train_one_epoch(train_loader, model_ft, optimizer, criterion, device, loss_w)
            # cam.activations_and_grads.release()  # 推理时把hooks给去掉
            val_iou, val_dice, val_acc = val_one_epoch(val_loader, model_ft, criterion, device, loss_w)
            print('train_iou = %.4f, train_dice = %.4f, train_acc = %.4f' % (train_iou, train_dice, train_acc))
            print('val_iou = %.4f, val_dice = %.4f, val_acc = %.4f' % (val_iou, val_dice, val_acc))
            scheduler.step()

            if loss_w[1] * val_iou + loss_w[0] * val_acc > loss_w[0] * best_index['acc'] + loss_w[1] * best_index['iou']:
                # best_acc = val_acc
                best_index = {'acc': val_acc, 'iou': val_iou}
                best_model_wts = copy.deepcopy(model_ft.state_dict())
                torch.save(model_ft.state_dict(),
                           os.path.join(save_dir,
                                        'wts_%d_%.4f__%.4f.pth' % (args.img_size, best_index['acc'], best_index['iou'])))

        torch.save(best_model_wts, os.path.join(save_dir, 'best_model_wts.pth'))

    elif args.mode == 'val':
        best_model_wts = torch.load(args.ckpt_path, map_location='cpu')
    
    '''5. start testing'''
    model_ft.load_state_dict(best_model_wts)
    device = torch.device(args.device)
    model_ft = model_ft.to(device)
    model_ft.eval()
    test_model(model_ft, test_loader, save_dir, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
