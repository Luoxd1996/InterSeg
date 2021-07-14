import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from medpy import metric
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataloader.dataset import (BraTS2018, RandomCrop,
                                RandomRotFlip, ReScale, ToTensor)
from loss import DiceLoss, dice_loss, ACELoss_3D
from networks.unet import UNet
from validation import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2018', help='Name of Experiment')
parser.add_argument('--max_iteration', type=int, default=30000,
                    help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,
                    default=0.01, help='learning rate')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
args = parser.parse_args()

patch_size = [96, 96, 96]
num_classes = 2

train_data_path = args.root_path
snapshot_path = "../model/BraTS2018/AutoSeg/"
batch_size = args.batch_size
max_iteration = args.max_iteration
base_lr = args.base_lr

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def cal_metric(gt, pred):
    gt, pred = gt.squeeze(0).cpu().data.numpy(
    ), pred.squeeze(0).cpu().data.numpy()
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        assd = metric.binary.assd(pred, gt)
        return np.array([dice, assd])
    else:
        return np.zeros(2)


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    net = UNet(in_chns=1, class_num=2).cuda()
    logging.info(net)

    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net)
    # net = net.cuda()

    db_train = BraTS2018(base_dir=train_data_path,
                         split='train',
                         full_num=None,
                         transform=transforms.Compose([
                             ReScale(patch_size),
                             RandomRotFlip(),
                             ToTensor(),
                         ]))

    db_test = BraTS2018(base_dir=train_data_path,
                        split='val',
                        transform=transforms.Compose([
                            ReScale(patch_size),
                            ToTensor(),
                        ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, num_workers=4)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr,
                          momentum=0.99, weight_decay=1e-4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    net.train()
    max_epoch = max_iteration // len(trainloader) + 1
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            print(volume_batch.shape, label_batch.shape)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = net(volume_batch)
            loss = F.cross_entropy(outputs, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iteration) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('lr/lr', lr_, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num > 0 and iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)
                outputs_soft = torch.softmax(outputs, dim=1)
                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 500 == 0:
                net.eval()
                metri = np.zeros(2)
                for index, sample in enumerate(testloader):
                    inputs, label = sample['image'].float(
                    ), sample['label'].int()
                    inputs, label = inputs.cuda(), label.cuda()
                    inputs, label = inputs, label
                    output = net(inputs)
                    output = torch.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1, keepdim=False)
                    metri += cal_metric(label, output)
                avg_metric = metri / len(testloader)
                if avg_metric[0] > best_performance:
                    best_performance = avg_metric[0]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_best_iter_{}.pth'.format(
                                                      iter_num))
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                writer.add_scalar('val/whole_tumor_dice',
                                  avg_metric[0], iter_num)
                writer.add_scalar('val/whole_tumor_assd',
                                  avg_metric[1], iter_num)
                net.train()

            if iter_num > 0 and iter_num % 2000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_{}.pth'.format(iter_num))
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iteration:
                break
            iter_num = iter_num + 1
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
