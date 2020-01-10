import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders.abus import ABUS, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler 
#from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.losses import dice_loss, boundary_loss, compute_sdf, threshold_loss, GeneralizedDiceLoss

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--root_path', type=str, default='../data/abus_roi/', help='Name of Experiment')
    parser.add_argument('--root_path', type=str, default='../data/abus_roi/', help='Name of Experiment')
    #parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')

    parser.add_argument('--max_iterations', type=int,  default=50000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')

    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=2019, help='random seed')
    parser.add_argument('--use_tm', action='store_true', default=False, help='whether use threshold_map')
    parser.add_argument('--use_dismap', action='store_true', default=True, help='whether use threshold_map')

    parser.add_argument('--save', type=str, default='../work/abus_roi/test')
    parser.add_argument('--writer_dir', type=str, default='../log/abus_roi/')
    args = parser.parse_args()

    return args

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

    #patch_size = (112, 112, 112)
    #patch_size = (160, 160, 160)
    patch_size = (64, 128, 128)
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

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True, use_tm=args.use_tm)
    net = net.cuda()

    #db_train = LAHeart(base_dir=train_data_path,
    #                   split='train',
    #                   transform = transforms.Compose([
    #                      RandomRotFlip(),
    #                      RandomCrop(patch_size),
    #                      ToTensor(),
    #                      ]))

    db_train = ABUS(base_dir=args.root_path,
                       split='train',
                       use_dismap=args.use_dismap,
                       transform = transforms.Compose([RandomRotFlip(use_dismap=args.use_dismap), RandomCrop(patch_size, use_dismap=args.use_dismap), ToTensor(use_dismap=args.use_dismap)]))
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #gdl = GeneralizedDiceLoss()

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
            volume_batch, label_batch, dis_map_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['dis_map']
            volume_batch, label_batch, dis_map_batch = volume_batch.cuda(), label_batch.cuda(), dis_map_batch.cuda()
            #print('volume_batch.shape: ', volume_batch.shape)
            if args.use_tm:
                outputs, tm = net(volume_batch)
                tm = torch.sigmoid(tm)
            else:
                outputs = net(volume_batch)
            #print('volume_batch.shape: ', volume_batch.shape)
            #print('outputs.shape, ', outputs.shape)

            loss_seg = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            #print(outputs_soft.shape)
            #print(label_batch.shape)
            #loss_seg_dice = gdl(outputs_soft, label_batch)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            #with torch.no_grad():
            #    # defalut using compute_sdf; however, compute_sdf1_1 is also worth to try;
            #    gt_sdf_npy = compute_sdf(label_batch.cpu().numpy(), outputs_soft.shape)
            #    gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs_soft.device.index)
            #    print('gt_sdf.shape: ', gt_sdf.shape)
            #loss_boundary = boundary_loss(outputs_soft, gt_sdf)

            #print('dis_map.shape: ', dis_map_batch.shape)
            loss_boundary = boundary_loss(outputs_soft, dis_map_batch)

            if args.use_tm:
                loss_threshold = threshold_loss(outputs_soft[:, 1, :, :, :], tm[:, 0, ...], label_batch == 1)
                loss_th = (0.1 * loss_seg + 0.9 * loss_seg_dice) + 3 * loss_threshold
                loss = alpha*(loss_th) + (1 - alpha) * loss_boundary
            else:
                loss = alpha * loss_seg_dice + (1-alpha) * loss_boundary

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out = outputs_soft.max(1)[1]
            dice = GeneralizedDiceLoss.dice_coeficient(out, label_batch)

            iter_num = iter_num + 1
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss_seg', loss_seg, iter_num)
            writer.add_scalar('train/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/alpha', alpha, iter_num)
            writer.add_scalar('train/loss', loss, iter_num)
            writer.add_scalar('train/dice', dice, iter_num)
            if args.use_tm:
                writer.add_scalar('train/loss_threshold', loss_threshold, iter_num)
            if args.use_dismap:
                writer.add_scalar('train/loss_dis', loss_boundary, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            logging.info('iteration %d : alpha : %f' % (iter_num, alpha))

            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, 30:71:10, :].permute(2,0,1,3)
                image = (image + 0.5) * 0.5
                grid_image = make_grid(image, 5)
                writer.add_image('train/Image', grid_image, iter_num)

                #outputs_soft = F.softmax(outputs, 1) #batchsize x num_classes x w x h x d
                image = outputs_soft[0, 1:2, :, 30:71:10, :].permute(2,0,1,3)
                grid_image = make_grid(image, 5, normalize=False)
                grid_image = grid_image.cpu().detach().numpy().transpose((1,2,0))

                gt = label_batch[0, :, 30:71:10, :].unsqueeze(0).permute(2,0,1,3)
                grid_gt = make_grid(gt, 5, normalize=False)
                grid_gt = grid_gt.cpu().detach().numpy().transpose((1,2,0))

                image_tm = dis_map_batch[0, :, :, 30:71:10, :].permute(2,0,1,3)
                #image_tm = tm[0, :, :, 30:71:10, :].permute(2,0,1,3)
                grid_tm = make_grid(image_tm, 5, normalize=False)
                grid_tm = grid_tm.cpu().detach().numpy().transpose((1,2,0))


                fig = plt.figure()
                ax = fig.add_subplot(311)
                ax.imshow(grid_gt[:, :, 0], 'gray')
                ax = fig.add_subplot(312)
                cs = ax.imshow(grid_image[:, :, 0], 'hot', vmin=0., vmax=1.)
                fig.colorbar(cs, ax=ax, shrink=0.9)
                ax = fig.add_subplot(313)
                cs = ax.imshow(grid_tm[:, :, 0], 'hot', vmin=0, vmax=1.)
                fig.colorbar(cs, ax=ax, shrink=0.9)
                writer.add_figure('train/prediction_results', fig, iter_num)
                fig.clear()

            ## change lr
            if iter_num % 5000 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 5000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        alpha -= 0.005
        if alpha <= 0.01:
            alpha = 0.01
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

if __name__ == "__main__":
    main()
