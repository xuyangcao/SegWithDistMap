import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from scipy.ndimage import distance_transform_edt as distance
from utils.losses import dice_loss

# Heart MR segmentation with hausdorff distance loss
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')

    parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')

    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=2019, help='random seed')

    parser.add_argument('--save', type=str, default='../work/la_heart/test')
    parser.add_argument('--writer_dir', type=str, default='../log/la_heart/')
    args = parser.parse_args()

    return args

def compute_dtm01(img_gt, out_shape):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """

    normalized_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis/np.max(posdis)

    return normalized_dtm

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:,1,...] - gt.float()) ** 2
    s_dtm = seg_dtm[:,1,...] ** 2
    g_dtm = gt_dtm[:,1,...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss

def main():
    ###################
    # init parameters #
    ###################
    args = get_args()
    # training path
    train_data_path = args.root_path
    # writer
    idx = args.save.rfind('/')
    log_dir = args.writer_dir + args.save[idx:]
    writer = SummaryWriter(log_dir)

    batch_size = args.batch_size * args.ngpu 
    max_iterations = args.max_iterations
    base_lr = args.base_lr

    patch_size = (112, 112, 80)
    num_classes = 2

    # random
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    ## make logger file
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    snapshot_path = args.save
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       num=16,
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    alpha = 1.0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            # volume_batch.shape=(b,1,x,y,z) label_patch.shape=(b,x,y,z)
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = net(volume_batch)

            loss_seg = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            # compute distance maps and hd loss
            with torch.no_grad():
                # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
                gt_dtm_npy = compute_dtm(label_batch.cpu().numpy(), outputs_soft.shape)
                gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
                seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
                seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)

            loss_hd = hd_loss(outputs_soft, label_batch, seg_dtm, gt_dtm)
            loss = alpha*(loss_seg+loss_seg_dice) + (1 - alpha) * loss_hd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_ce', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hd', loss_hd, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/alpha', alpha, iter_num)
            logging.info('iteration %d : alpha : %f' % (iter_num, alpha))
            logging.info('iteration %d : loss_dice : %f' % (iter_num, loss_seg_dice.item()))
            logging.info('iteration %d : loss_hd : %f' % (iter_num, loss_hd.item()))
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 2 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = gt_dtm[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/gt_sdf', grid_image, iter_num)

                image = seg_dtm[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/seg_sdf', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        alpha -= 0.001
        if alpha <= 0.001:
            alpha = 0.001
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

if __name__ == "__main__":
    main()
