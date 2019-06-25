#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, absolute_import, division

# import sys
# sys.setdefaultencoding("utf-8")

import os
import sys
import time
from pprint import pprint
import numpy as np
from tqdm import trange
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
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
from src import Bar
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


    ## stat_3d.keys() =>  dict_keys(['std', 'dim_use', 'train', 'test', 'mean'])
    ## std => (96., )
    ## mean => (96.,)
    ## dim_use => (48, ) ?????
    # train => dict{[user, action, camera_id]} ex) dict{[6, 'Walking', 'Walking 1.60457274.h5']} // data = int // len 600 = 15 actions * 8 cameras+extra_actions * 5 users
    # test => same as train, user = 9, 11 // len 240
    # (7,
    #  'Photo',
    #  'Photo 1.58860488.h5'): array([[514.54570615, -606.40670751, 5283.29114444],
    #                                 [513.19690503, -606.27874917, 5282.94296128],
    #                                 [511.72623278, -606.3556718, 5282.09161439],
    #                                 ...,
    #                                 [660.21544235, -494.87670603, 5111.48298849],
    #                                 [654.79473179, -497.67942449, 5111.05843265],
    #                                 [649.61962945, -498.74291164, 5111.91590807]])}


    # test
    if opt.test:
        err_set = []
        for action in actions:
            print (">>> TEST on _{}_".format(action))

            test_loader = DataLoader(
                dataset=Human36M(actions=action, data_path=opt.data_dir, use_hg=opt.use_hg, is_train=False),
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
        dataset=Human36M(actions=actions, data_path=opt.data_dir, use_hg=opt.use_hg, is_train=False),
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)

    train_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, use_hg=opt.use_hg),
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
            train_loader, model, criterion, optimizer,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        # test
        loss_test, err_test = test(test_loader, model, criterion, stat_3d, procrustes=opt.procrustes)
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


def train(train_loader, model, criterion, optimizer,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):

    losses = utils.AverageMeter()

    model.train()

    # start = time.time()
    # batch_time = 0
    # bar = Bar('>>>', fill='>', max=len(train_loader))

    # for i, (inps, tars) in enumerate(train_loader): # inps = (64, 32)
    pbar = tqdm(train_loader)
    for i, (inps, tars) in enumerate(pbar): # inps = (64, 32)
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

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




    #     # update summary
    #     if (i + 1) % 100 == 0:
    #         batch_time = time.time() - start
    #         start = time.time()
    #
    #     bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
    #         .format(batch=i + 1,
    #                 size=len(train_loader),
    #                 batchtime=batch_time * 10.0,
    #                 ttl=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg)
    #     bar.next()
    #
    # bar.finish()

    return glob_step, lr_now, losses.avg

# def test(test_loader, model, criterion, stat2d, stat_3d, procrustes=False):

def test(test_loader, model, criterion, stat_3d, procrustes=False):

    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
    # start = time.time()
    # batch_time = 0
    # bar = Bar('>>>', fill='>', max=len(test_loader))

    # for i, (inps, tars) in enumerate(test_loader):

    pbar = tqdm(test_loader)
    for i, (inps, tars) in enumerate(pbar):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        outputs_coord = outputs
        loss = criterion(outputs_coord, targets) # 64 losses average

        losses.update(loss.item(), inputs.size(0))

        tars = targets

        # inputs_unnorm = data_process.unNormalizeData(inputs.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
        # calculate erruracy
        targets_unnorm = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        outputs_unnorm = data_process.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        # remove dim ignored
        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))
        # dim_2d_use = stat_2d['dim_use']

        # stat_3d['dim_use'] (48., )
        outputs_use = outputs_unnorm[:, dim_use]
        targets_use = targets_unnorm[:, dim_use] # (51, )
        # inputs_use = inputs_unnorm[:, dim_2d_use] # (51, )

        if procrustes:
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3) # (17,3)
                _, Z, T, b, c = get_transformation(gt, out, True)
                out = (b * out.dot(T)) + c
                outputs_use[ba, :] = out.reshape(1, 51)

        sqerr = (outputs_use - targets_use) ** 2

        distance = np.zeros((sqerr.shape[0], 17))
        dist_idx = 0
        for k in np.arange(0, 17 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)

        # # update summary
        # if (i + 1) % 100 == 0:
        #     batch_time = time.time() - start
        #     start = time.time()
        #
        # bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
        #     .format(batch=i + 1,
        #             size=len(test_loader),
        #             batchtime=batch_time * 10.0,
        #             ttl=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg)
        # bar.next()
        pbar.set_postfix(tt_loss='{:05.6f}'.format(losses.avg))


    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    # bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


if __name__ == "__main__":
    option = Options().parse()
    main(option)
    print(main)


