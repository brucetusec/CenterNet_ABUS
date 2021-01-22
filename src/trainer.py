import os, sys, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#from apex import amp
from models.networks.hourglass import get_large_hourglass_net
from models.loss import FocalLoss, RegL1Loss, RegL2Loss
from data.abus_data import AbusNpyFormat
import math
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()

import logging

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class SimpleCheckpointer(object):
    def __init__(self, checkpoint_dir, model):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


    def load(self, ep):
        #path = path + self.model_name + '_' + str(ep)
        path = self.checkpoint_dir + str(ep)
        self.model.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        e.g. AlexNet_0710_23:57:29.pth
        '''
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, name + '.pth'))
        return name

def train(args):
    checkpoint_dir = 'checkpoints/{}'.format(args.exp_name)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    logger = setup_logger("CenterNet_ABUS", checkpoint_dir, distributed_rank=0)
    logger.info(args)

    logger.info('Preparing...')
    validset = AbusNpyFormat(testing_mode=0, root=root, crx_valid=True, crx_fold_num=args.crx_valid, crx_partition='valid', augmentation=False)
    trainset = AbusNpyFormat(testing_mode=0, root=root, crx_valid=True, crx_fold_num=args.crx_valid, crx_partition='train', augmentation=True)
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    validset_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=6)

    crit_hm = FocalLoss()
    crit_wh = RegL1Loss()

    train_hist = {
        'train_loss':[],
        'valid_hm_loss':[],
        'valid_wh_loss':[],
        'valid_total_loss':[],
        'per_epoch_time':[]
    }

    heads = {
        'hm': 1,
        'wh': 3
    }
    model = get_large_hourglass_net(heads, n_stacks=1)
    model = model.to(device)
    checkpointer = SimpleCheckpointer(checkpoint_dir, model)
    if args.resume:
        init_ep = 0
        logger.info('Resume training from the designated checkpoint.')
        checkpointer.load(str(args.resume_ep))
    else:
        init_ep = 0
    end_ep = args.max_epoch

    if args.freeze:
        logger.info('Paritially freeze layers.')
        for param in model.pre.parameters():
            param.requires_grad = False
        for param in model.kps.parameters():
            param.requires_grad = False
        for param in model.cnvs.parameters():
            param.requires_grad = False
        for param in model.inters.parameters():
            param.requires_grad = False
        for param in model.inters_.parameters():
            param.requires_grad = False
        for param in model.cnvs_.parameters():
            param.requires_grad = False
        for param in model.hm.parameters():
            param.requires_grad = False
        crit_wh = RegL2Loss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optim_sched = ExponentialLR(optimizer, 0.95, last_epoch=-1)

    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    logger.info('Preparation done.')
    logger.info('******************')
    logger.info('Training starts...')

    start_time = time.time()
    min_loss = 0

    checkpointer.save('initial')
    first_ep = True

    for epoch in range(init_ep, end_ep):
        epoch_start_time = time.time()
        train_loss = 0
        current_loss = 0
        valid_hm_loss = 0
        valid_wh_loss = 0
        lambda_s = args.lambda_s # * (1.03**epoch)

        # Training
        model.train()
        optimizer.zero_grad()
        for batch_idx, (data_img, data_hm, data_wh, _) in enumerate(trainset_loader):
            if use_cuda:
                data_img = data_img.cuda()
                data_hm = data_hm.cuda()
                data_wh = data_wh.cuda()
            output = model(data_img)
            hm_loss = crit_hm(output[-1]['hm'], data_hm)
            wh_loss = crit_wh(output[-1]['wh'], data_wh)

            total_loss = hm_loss + lambda_s*wh_loss
            train_loss += (hm_loss.item() + args.lambda_s*wh_loss.item())
            total_loss.backward()
            if  (first_ep and batch_idx < 10) or ((batch_idx % 16) is 0) or (batch_idx == len(trainset_loader) - 1):
                logger.info('Gradient applied at batch #{}  '.format(batch_idx))
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch: [{:2d}] [{:3d}], hm_loss: {:.3f}, wh_loss: {:.3f}, total_loss: {:.3f}"\
                .format((epoch + 1), (batch_idx + 1), hm_loss.item(), wh_loss.item(), total_loss.item()))

        optim_sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch_idx, (data_img, data_hm, data_wh, _) in enumerate(validset_loader):
                if use_cuda:
                    data_img = data_img.cuda()
                    data_hm = data_hm.cuda()
                    data_wh = data_wh.cuda()
                output = model(data_img)
                hm_loss = crit_hm(output[-1]['hm'], data_hm)
                wh_loss = crit_wh(output[-1]['wh'], data_wh)

                valid_hm_loss += hm_loss.item()
                valid_wh_loss += wh_loss.item()

        valid_hm_loss = valid_hm_loss/validset.__len__()
        valid_wh_loss = valid_wh_loss/validset.__len__()
        train_loss = train_loss/trainset.__len__()
        current_loss = valid_hm_loss + args.lambda_s*valid_wh_loss

        save_id = (args.resume_ep + '_' + str(epoch)) if args.resume else str(epoch)
        if epoch == 0 or current_loss < min_loss:
            min_loss = current_loss
            checkpointer.save(save_id)
        elif (epoch % 5) == 4:
            checkpointer.save(save_id)
        checkpointer.save('latest')

        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        train_hist['valid_hm_loss'].append(valid_hm_loss)
        train_hist['valid_wh_loss'].append(args.lambda_s*valid_wh_loss)
        train_hist['valid_total_loss'].append(current_loss)
        train_hist['train_loss'].append(train_loss)
        plt.figure()
        plt.plot(train_hist['train_loss'], color='k')
        plt.plot(train_hist['valid_total_loss'], color='r')
        plt.plot(train_hist['valid_hm_loss'], color='b')
        plt.plot(train_hist['valid_wh_loss'], color='g')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig('loss_fold{}.png'.format(args.crx_valid))
        plt.close()
        np.save('train_hist_{}.npy'.format(args.exp_name), train_hist)
        logger.info("Epoch: [{:d}], valid_hm_loss: {:.3f}, valid_wh_loss: {:.3f}".format((epoch + 1), valid_hm_loss, args.lambda_s*valid_wh_loss))
        logger.info('Epoch exec time: {} min'.format((time.time() - epoch_start_time)/60))
        first_ep = False

    logger.info("Training finished.")
    logger.info("Total time cost: {} min.".format((time.time() - start_time)/60))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', nargs='?', type=str, default='debug',
        help='experiment name to saved tensorboard log')
    parser.add_argument('--crx_valid', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lambda_s', type=float, default=0.1)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--resume_ep', type=str, default='0')
    parser.add_argument('--freeze', dest='freeze', action='store_true')
    parser.set_defaults(resume=False)
    parser.set_defaults(freeze=False)
    return parser.parse_args()

if __name__ == '__main__':
    if use_cuda:
        print('GPU is available.')

    args = _parse_args()
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    root = 'data/sys_ucc/'


    train(args)