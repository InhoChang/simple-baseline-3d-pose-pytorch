#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, absolute_import, division

# import sys
# sys.setdefaultencoding("utf-8")

import os
import sys
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opt import Options
from src.procrustes import get_transformation
import src.data_process as data_process
import src.utils as utils
import src.misc as misc
import src.log as log

from src.model import LinearModel, weight_init
from src.datasets.human36m import Human36M


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = LinearModel()
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:

        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load, encoding='utf-8')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    # list of action(s)
    actions = misc.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    # pprint(actions, indent=4)
    # print(">>>")

    # data loading
    print(">>> loading data")
    # load statistics data
    stat_3d = torch.load(os.path.join(opt.data_dir, 'stat_3d.pth.tar'))
    stat_2d = torch.load(os.path.join(opt.data_dir, 'stat_2d.pth.tar'))

    # test
    if opt.test:
        err_set = []
        for action in actions:
            print (">>> TEST on _{}_".format(action))

            test_loader = DataLoader(
                dataset=Human36M(actions=action, data_path=opt.data_dir,set_num_samples = opt.set_num_samples, use_hg=opt.use_hg, is_train=False),
                batch_size=opt.test_batch,
                shuffle=False,
                num_workers=opt.job,
                pin_memory=True)

            _, err_test = test(test_loader, model, criterion, stat_2d, stat_3d, procrustes=opt.procrustes)
            err_set.append(err_test)

        print (">>>>>> TEST results:")

        for action in actions:
            print ("{}".format(action), end='\t')
        print ("\n")

        for err in err_set:
            print ("{:.4f}".format(err), end='\t')
        print (">>>\nERRORS: {}".format(np.array(err_set).mean()))
        sys.exit()

    # load datasets for training
    test_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, set_num_samples = opt.set_num_samples, use_hg=opt.use_hg, is_train=False),
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)

    train_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, set_num_samples = opt.set_num_samples, use_hg=opt.use_hg),
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)

    print(">>> data loaded !")

    cudnn.benchmark = True

    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        ## per epoch
        # train
        glob_step, lr_now, loss_train = train(
            train_loader, model, criterion, optimizer, stat_2d, stat_3d,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        # test
        loss_test, err_test = test(test_loader, model, criterion, stat_2d, stat_3d, procrustes=opt.procrustes)
        # loss_test, err_test = test(test_loader, model, criterion, stat_3d, procrustes=True)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, loss_test, err_test],
                      ['int', 'float', 'float', 'float', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)

        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=False)

    logger.close()


def train(train_loader, model, criterion, optimizer, stat_2d, stat_3d,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True ):

    losses = utils.AverageMeter()

    model.train()


    # for i, (inps, tars) in enumerate(train_loader): # inps = (64, 32)
    pbar = tqdm(train_loader)
    for i, (inps, tars) in enumerate(pbar): # inps = (64, 32)
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        ### Input unnormalization
        inputs_unnorm = data_process.unNormalizeData(inps.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use']) # 64, 64
        dim_2d_use = stat_2d['dim_use']
        inputs_use = inputs_unnorm[:, dim_2d_use]  # (64, 32)
        ### Input distance normalization
        inputs_dist_norm, _ = data_process.input_norm(inputs_use) # (64, 32) , array
        input_dist = torch.tensor(inputs_dist_norm, dtype=torch.float32)

        ### Targets unnormalization
        targets_unnorm = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use']) # (64, 96)
        dim_3d_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22,
                               23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58,
                               59, 75, 76, 77, 78, 79, 80, 81, 82, 83])
        targets_use = targets_unnorm[:, dim_3d_use] # (51, )

        ### Targets distance normalization
        targets_dist_norm, _  = data_process.output_norm(targets_use)
        targets_dist = torch.tensor(targets_dist_norm, dtype=torch.float32)

        inputs = Variable(input_dist.cuda())
        targets = Variable(targets_dist.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        loss.backward()

        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # tqdm.set_postfix(loss='{:05.6f}'.format(losses.avg))
        pbar.set_postfix(tr_loss='{:05.6f}'.format(losses.avg))


    return glob_step, lr_now, losses.avg

# def test(test_loader, model, criterion, stat2d, stat_3d, procrustes=False):

def test(test_loader, model, criterion, stat_2d, stat_3d, procrustes=False):

    losses = utils.AverageMeter()

    model.eval()

    all_dist = []

    pbar = tqdm(test_loader)
    for i, (inps, tars) in enumerate(pbar):

        ### input unnorm
        data_coord = data_process.unNormalizeData(inps.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use']) # 64, 64
        dim_2d_use = stat_2d['dim_use']
        data_use = data_coord[:, dim_2d_use]  # (64, 32)

        ### input dist norm
        data_dist_norm, data_dist_set = data_process.input_norm(data_use) # (64, 32) , array
        data_dist = torch.tensor(data_dist_norm, dtype=torch.float32)

        # target unnorm
        label_coord = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use']) # (64, 96)
        dim_3d_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22,
                               23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58,
                               59, 75, 76, 77, 78, 79, 80, 81, 82, 83])

        label_use = label_coord[:, dim_3d_use]  # (48, )
        # target dist norm
        label_dist_norm, label_dist_set = data_process.output_norm(label_use)
        label_dist = torch.tensor(label_dist_norm, dtype=torch.float32)

        inputs = Variable(data_dist.cuda())
        targets = Variable(label_dist.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        pred_coord = outputs
        loss = criterion(pred_coord, targets) # 64 losses average

        losses.update(loss.item(), inputs.size(0))

        tars = targets
        pred = outputs

        # inputs_dist_set = np.reshape(targets_dist_set, (-1, 1))
        # inputs_dist_set = np.repeat(targets_dist_set, 48, axis=1)

        targets_dist = np.reshape(label_dist_set, (-1, 1))
        targets_dist_set = np.repeat(targets_dist, 48, axis=1)

        c = np.reshape( np.asarray( [0,0,10]) , (1,-1) )
        c = np.repeat(c, 16, axis=0)
        c = np.reshape(c, (1, -1) )
        c = np.repeat(c, inputs.size(0) , axis=0)
        # c_set = np.repeat(np.asarray([0,0,10]), 16, axis=0)

        #### undist -> unnorm
        outputs_undist = (pred.data.cpu().numpy() * targets_dist_set) - c
        # outputs_undist = outputs_undist - c
        targets_undist =  ( tars.data.cpu().numpy() * targets_dist_set ) - c
        # targets_undist = targets_undist - c


        outputs_use = outputs_undist
        targets_use = targets_undist# (64, 48)

        if procrustes:
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3) # (17,3)
                _, Z, T, b, c = get_transformation(gt, out, True)
                out = (b * out.dot(T)) + c
                outputs_use[ba, :] = out.reshape(1, 48)

        sqerr = (outputs_use - targets_use) ** 2

        # distance = np.zeros((sqerr.shape[0], 17))
        distance = np.zeros((sqerr.shape[0], 16))

        dist_idx = 0
        for k in np.arange(0, 16 * 3, 3):
        # for k in np.arange(0, 17 * 3, 3):

            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)



        pbar.set_postfix(tt_loss='{:05.6f}'.format(losses.avg))


    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    # bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


if __name__ == "__main__":
    option = Options().parse()
    option.set_num_samples = -1
    option.procrustes = False
    option.test = False
    option.resume = False # If you want to resume train from previous ckpt then set as True. Also, have to set option.load file path
    # option.load = 'D:\\Workspace\\3d_pose_baseline_pytorch-master\\3d_pose_baseline_pytorch-master\\checkpoint\\test\\ckpt_best.pth.tar' # file_path where ckpt files are in
    option.load = '' # file_path where ckpt files are in


    main(option)
    # print(main)


