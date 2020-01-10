##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time, time_string_short
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_102_api  import NASBench102API as API
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchviz import make_dot
from models.cell_searchs.genotypes import AllFull_CODE,AllConv3x3_CODE, AllConv1x1_CODE


def op_list2str(op_list):
    search_space=['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    best_arch_nums=''
    print(op_list)
    for ops in op_list:
        for op in ops:
            best_arch_nums += str(search_space.index(op[0]))
    return best_arch_nums


def pre_train_shared_cnn(xloader, yloader, shared_cnn, controller, criterion, scheduler, optimizer, print_freq=100):
    data_time, batch_time = AverageMeter(), AverageMeter()
    losses, top1s, top5s, xend = AverageMeter(), AverageMeter(), AverageMeter(), time.time()
    sampled_arch = AllConv3x3_CODE

    shared_cnn.module.update_arch(sampled_arch)

    for epoch in range(10):
        shared_cnn.train()
        controller.eval()
        for step, (inputs, targets) in enumerate(xloader):
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            _, logits = shared_cnn(inputs)
            loss      = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), 5)
            optimizer.step()

            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 2))

            losses.update(loss.item(),  inputs.size(0))

            # print('1', logits.data[0], targets.data[0], prec1)

            # measure elapsed time
            # batch_time.update(time.time() - xend)
            # xend = time.time()
            top1s.update(prec1.item(), inputs.size(0))
            top5s.update(prec5.item(), inputs.size(0))

        Sstr = '*Train-Shared-CNN* ' + time_string() + ' [{:}]'.format(str(epoch))
        Wstr = 'acc@1 {top1.avg:.2f} Prec@2 {top5.avg:.2f}]'.format(top1=top1s, top5=top5s)
        print(Sstr + ' ' + Wstr)
        top1s.reset()
        top5s.reset()

        for step, (inputs, targets) in enumerate(yloader):
            _, logits = shared_cnn(inputs)

            val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 2))
            top1s.update (val_top1.item(), inputs.size(0))
            top5s.update (val_top5.item(), inputs.size(0))

            # print('2',logits.data[0], targets.data[0],val_top1)



        Sstr = '*val-Shared-CNN* ' + time_string() + ' [{:}]'.format(str(epoch))
        Wstr = 'acc@1 {top1.avg:.2f} Prec@2 {top5.avg:.2f}]'.format(top1=top1s, top5=top5s)
        print(Sstr +  ' ' + Wstr)
        top1s.reset()
        top5s.reset()
    return losses.avg, top1s.avg, top5s.avg


def train_shared_cnn(xloader, shared_cnn, controller, criterion, scheduler, optimizer, epoch_str, print_freq, logger):
    data_time, batch_time = AverageMeter(), AverageMeter()
    losses, top1s, top5s, xend = AverageMeter(), AverageMeter(), AverageMeter(), time.time()

    shared_cnn.train()
    controller.eval()
    ne = 10

    for ni in range(ne):
        with torch.no_grad():
            _, _, sampled_arch = controller()
        shared_cnn.module.update_arch(sampled_arch)
        print(sampled_arch)
        # arch_str = op_list2str(sampled_arch)
        for step, (inputs, targets) in enumerate(xloader):
            # print(step,inputs,targets)
            scheduler.update(None, 1.0 * step / len(xloader))
            targets = targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - xend)



            optimizer.zero_grad()

            _, logits = shared_cnn(inputs)
            loss      = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), 5)
            optimizer.step()
            # record
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 2))
            losses.update(loss.item(),  inputs.size(0))
            top1s.update (prec1.item(), inputs.size(0))
            top5s.update (prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - xend)
            xend = time.time()

            # if step + 1 == len(xloader):
        Sstr = '*Train-Shared-CNN* ' + time_string() + ' [{:03d}/10]'.format(ni, ne)
        Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
        Wstr = '[Loss {loss.avg:.3f}  Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f}]'.format(loss=losses, top1=top1s, top5=top5s)
        losses.reset()
        top1s.reset()
        top5s.reset()
        logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)

    return losses.avg, top1s.avg, top5s.avg


def train_controller(xloader, shared_cnn, controller, criterion, optimizer, config, epoch_str, print_freq, logger):
    # config. (containing some necessary arg)
    #   baseline: The baseline score (i.e. average val_acc) from the previous epoch
    data_time, batch_time = AverageMeter(), AverageMeter()
    GradnormMeter, LossMeter, ValAccMeter, EntropyMeter, BaselineMeter, RewardMeter, xend = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()

    shared_cnn.eval()
    controller.train()
    controller.zero_grad()
    #for step, (inputs, targets) in enumerate(xloader):
    loader_iter = iter(xloader)
    for step in range(config.ctl_train_steps * config.ctl_num_aggre):
        try:
            inputs, targets = next(loader_iter)
        except:
            loader_iter = iter(xloader)
            inputs, targets = next(loader_iter)
        targets = targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - xend)

        log_prob, entropy, sampled_arch = controller()
        with torch.no_grad():
            shared_cnn.module.update_arch(sampled_arch)
            _, logits = shared_cnn(inputs)
            val_top1, val_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 2))
            val_top1  = val_top1.view(-1) / 100
        reward = val_top1 + config.ctl_entropy_w * entropy
        if config.baseline is None:
            baseline = val_top1
        else:
            baseline = config.baseline - (1 - config.ctl_bl_dec) * (config.baseline - reward)

        loss = 1 * log_prob * (reward - baseline)

        # account
        RewardMeter.update(reward.item())
        BaselineMeter.update(baseline.item())
        ValAccMeter.update(val_top1.item()*100)
        LossMeter.update(loss.item())
        EntropyMeter.update(entropy.item())

        # Average gradient over controller_num_aggregate samples
        loss = loss / config.ctl_num_aggre
        loss.backward(retain_graph=True)

        # measure elapsed time
        batch_time.update(time.time() - xend)
        xend = time.time()
        if (step+1) % config.ctl_num_aggre == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
            GradnormMeter.update(grad_norm)
            optimizer.step()
            controller.zero_grad()

        # if step % print_freq == 0:
        #     Sstr = '*Train-Controller* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, config.ctl_train_steps * config.ctl_num_aggre)
        #     Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
        #     Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Reward {reward.val:.2f} ({reward.avg:.2f})] Baseline {basel.val:.2f} ({basel.avg:.2f})'.format(loss=LossMeter, top1=ValAccMeter, reward=RewardMeter, basel=BaselineMeter)
        #     Estr = 'Entropy={:.4f} ({:.4f})'.format(EntropyMeter.val, EntropyMeter.avg)
        #     logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Estr)

    return LossMeter.avg, ValAccMeter.avg, BaselineMeter.avg, RewardMeter.avg, baseline.item()


def get_best_arch(controller, shared_cnn, xloader, n_samples=10):
    # with torch.no_grad():
    controller.eval()
    # shared_cnn.eval()
    shared_cnn.train()
    archs, valid_accs = [], []
    loader_iter = iter(xloader)
    for i in range(n_samples):
        try:
            inputs, targets = next(loader_iter)
        except:
            loader_iter = iter(xloader)
            inputs, targets = next(loader_iter)

        _, _, sampled_arch = controller()
        arch = shared_cnn.module.update_arch(sampled_arch)
        _, logits = shared_cnn(inputs)
        val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 2))

        archs.append( arch )
        valid_accs.append( val_top1.item() )
        print(arch,val_top1)

    best_idx = np.argmax(valid_accs)
    best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
    return best_arch, best_valid_acc


def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.eval()
    end = time.time()
    with torch.no_grad():
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 2))
            arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
            arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads( xargs.workers )
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    train_data, test_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
    print(xshape)
    xshape=(1,3,88,88)
    class_num = 4
    if xargs.dataset == 'cifar10' or xargs.dataset == 'cifar100':
        split_Fpath = '/home/city/Projects/NAS-Projects/configs/nas-benchmark/cifar-split.txt'
        cifar_split = load_config(split_Fpath, None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid
        logger.log('Load split file from {:}'.format(split_Fpath))
    # elif xargs.dataset.startswith('ImageNet16'):
    #   split_Fpath = '/home/city/Projects/NAS-Projects/configs/nas-benchmark/{:}-split.txt'.format(xargs.dataset)
    #   imagenet16_split = load_config(split_Fpath, None, None)
    #   train_split, valid_split = imagenet16_split.train, imagenet16_split.valid
    #   logger.log('Load split file from {:}'.format(split_Fpath))
    # else:
    #   raise ValueError('invalid dataset : {:}'.format(xargs.dataset))
    logger.log('use config from : {:}'.format(xargs.config_path))
    config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
    logger.log('config: {:}'.format(config))
    # To split data
    train_data_v2 = deepcopy(train_data)
    train_data_v2.transform = test_data.transform
    valid_data    = train_data_v2

    train_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[128, 128, 128], std=[50, 50, 50])
    ])
    val_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[128, 128, 128], std=[50, 50, 50])
    ])

    train_data = datasets.ImageFolder(root='/home/city/Projects/build_assessment/data/train',
                                      transform=train_transform)
    valid_data = datasets.ImageFolder(root='/home/city/Projects/build_assessment/data/val',
                                      transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                                 batch_size=32, shuffle=True,
                                                 num_workers=4
                                                 )
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                                 batch_size=32, shuffle=True,
                                                 num_workers=2
                                                 )
    # data loader
    # train_loader  = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split), num_workers=xargs.workers, pin_memory=True)
    # valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=xargs.workers, pin_memory=True)
    logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), len(valid_loader), config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    print('search space', search_space)
    model_config = dict2config({'name': 'ENAS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space}, None)
    shared_cnn = get_cell_based_tiny_net(model_config)
    # input_data = Variable(torch.rand(16, 3, 224, 224))
    # print(shared_cnn)
    # writer = SummaryWriter(log_dir='/home/city/disk/tmp/log', comment='resnet18')
    # with writer:
    #   writer.add_graph(shared_cnn, (input_data,))
    # exit()
    print(len(shared_cnn.edge2index), len(shared_cnn.op_names), 'lend edge2index,len op names')
    print(shared_cnn.edge2index)
    print(shared_cnn.op_names)
    controller = shared_cnn.create_controller()
    print(controller)

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(shared_cnn.parameters(), config)
    a_optimizer = torch.optim.Adam(controller.parameters(), lr=config.controller_lr, betas=config.controller_betas, eps=config.controller_eps)
    logger.log('w-optimizer : {:}'.format(w_optimizer))
    logger.log('a-optimizer : {:}'.format(a_optimizer))
    logger.log('w-scheduler : {:}'.format(w_scheduler))
    logger.log('criterion   : {:}'.format(criterion))
    #flop, param  = get_model_infos(shared_cnn, xshape)
    #logger.log('{:}'.format(shared_cnn))
    #logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space : {:}'.format(search_space))
    # if xargs.arch_nas_dataset is None:
    #     #     api = None
    #     # else:
    #     #     api = API(xargs.arch_nas_dataset)
    api = None

    logger.log('{:} create API = {:} done'.format(time_string(), api))
    shared_cnn, controller, criterion = torch.nn.DataParallel(shared_cnn).cuda(), controller.cuda(), criterion.cuda()


    #tw.model_stats(shared_cnn, [13, 3, 224, 224])
    last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')

    if last_info.exists(): # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info   = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint  = torch.load(last_info['last_checkpoint'])
        genotypes   = checkpoint['genotypes']
        baseline    = checkpoint['baseline']
        valid_accuracies = checkpoint['valid_accuracies']
        shared_cnn.load_state_dict( checkpoint['shared_cnn'] )
        controller.load_state_dict( checkpoint['controller'] )
        w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
        w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
        a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
        logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes, baseline = 0, {'best': -1}, {}, None

    pre_train_shared_cnn(train_loader, valid_loader, shared_cnn, controller, criterion, w_scheduler, w_optimizer)
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(shared_cnn.parameters(), config)
    # start training
    start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        logger.log('\n[Search the {:}-th epoch] {:}, LR={:}, baseline={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr()), baseline))

        cnn_loss, cnn_top1, cnn_top5 = train_shared_cnn(train_loader, shared_cnn, controller, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger)
        logger.log('[{:}] shared-cnn : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, cnn_loss, cnn_top1, cnn_top5))
        ctl_loss, ctl_acc, ctl_baseline, ctl_reward, baseline \
            = train_controller(valid_loader, shared_cnn, controller, criterion, a_optimizer, \
                               dict2config({'baseline': baseline,
                                            'ctl_train_steps': xargs.controller_train_steps, 'ctl_num_aggre': xargs.controller_num_aggregate,
                                            'ctl_entropy_w': xargs.controller_entropy_weight,
                                            'ctl_bl_dec'   : xargs.controller_bl_dec}, None), \
                               epoch_str, xargs.print_freq, logger)
        search_time.update(time.time() - start_time)
        logger.log('[{:}] controller : loss={:.2f}, accuracy={:.2f}%, baseline={:.2f}, reward={:.2f}, current-baseline={:.4f}, time-cost={:.1f} s'.format(epoch_str, ctl_loss, ctl_acc, ctl_baseline, ctl_reward, baseline, search_time.sum))
        best_arch, _ = get_best_arch(controller, shared_cnn, valid_loader)
        shared_cnn.module.update_arch(best_arch)
        _, best_valid_acc, _ = valid_func(valid_loader, shared_cnn, criterion)
        op_list,_ = best_arch.tolist(remove_str=None)
        best_arch_nums=op_list2str(op_list)

        genotypes[epoch] = best_arch
        # check the best accuracy
        valid_accuracies[epoch] = best_valid_acc

        if best_valid_acc > valid_accuracies['best']:
            valid_accuracies['best'] = best_valid_acc
            genotypes['best']        = best_arch
            find_best = True
            torch.save(shared_cnn,'/home/city/disk/log/builds-enas/share_cnn_%s_%s_%.2f.pth' % (time_string_short(),best_arch_nums,best_valid_acc))
            torch.save(controller, '/home/city/disk/log/builds-enas/controller_%s_%s_%.2f.pth' % (time_string_short(),best_arch_nums,best_valid_acc))
            print('/home/city/disk/log/builds-enas/share_cnn_%s_%s_%.2f.pth' % (time_string_short(),best_arch_nums,best_valid_acc))


        else: find_best = False
        print(best_valid_acc, valid_accuracies['best'])
        logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
        # save checkpoint
        save_path = save_checkpoint({'epoch' : epoch + 1,
                                     'args'  : deepcopy(xargs),
                                     'baseline'    : baseline,
                                     'shared_cnn'  : shared_cnn.state_dict(),
                                     'controller'  : controller.state_dict(),
                                     'w_optimizer' : w_optimizer.state_dict(),
                                     'a_optimizer' : a_optimizer.state_dict(),
                                     'w_scheduler' : w_scheduler.state_dict(),
                                     'genotypes'   : genotypes,
                                     'valid_accuracies' : valid_accuracies},
                                    model_base_path, logger)
        last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args' : deepcopy(args),
            'last_checkpoint': save_path,
        }, logger.path('info'), logger)
        if find_best:
            logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, best_valid_acc))
            copy_checkpoint(model_base_path, model_best_path, logger)
        if api is not None: logger.log('{:}'.format(api.query_by_arch( genotypes[epoch] )))
        # measure elapsed time

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        # with open("/home/city/disk/tmp/foo.txt") as file:
        #   in_x = torch.rand(13, 3, 128, 128)
        #   # in_x =
        #   g = make_dot(shared_cnn(in_x), params=dict(list(shared_cnn.named_parameters()) + [('x', in_x)]))
        #   g.render('/home/city/disk/tmp/espnet_model_%04d' % epoch, view=False)
        # # g = None

    logger.log('\n' + '-'*100)
    logger.log('During searching, the best architecture is {:}'.format(genotypes['best']))
    logger.log('Its accuracy is {:.2f}%'.format(valid_accuracies['best']))
    logger.log('Randomly select {:} architectures and select the best.'.format(xargs.controller_num_samples))
    start_time = time.time()
    final_arch, _ = get_best_arch(controller, shared_cnn, valid_loader, xargs.controller_num_samples)
    search_time.update(time.time() - start_time)
    shared_cnn.module.update_arch(final_arch)
    final_loss, final_top1, final_top5 = valid_func(valid_loader, shared_cnn, criterion)
    logger.log('The Selected Final Architecture : {:}'.format(final_arch))
    logger.log('Loss={:.3f}, Accuracy@1={:.2f}%, Accuracy@5={:.2f}%'.format(final_loss, final_top1, final_top5))
    logger.log('ENAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum, final_arch))
    if api is not None: logger.log('{:}'.format( api.query_by_arch(final_arch) ))
    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("ENAS")
    parser.add_argument('--data_path',          type=str,   default="/home/city/.torch/cifar.python",     help='Path to dataset')
    parser.add_argument('--dataset',            type=str,   default='cifar10',     choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
    # channels and number-of-cells
    parser.add_argument('--search_space_name',  type=str,   default='nas-bench-102',   help='The search space name.')
    parser.add_argument('--max_nodes',          type=int,   default='4',   help='The maximum number of nodes.')
    parser.add_argument('--channel',            type=int,   default='16',   help='The number of channels.')
    parser.add_argument('--num_cells',          type=int,   default='1',   help='The number of cells in one stage.')
    parser.add_argument('--config_path',        type=str,   default='/home/city/Projects/NAS-Projects/configs/nas-benchmark/algos/ENAS.config',   help='The config file to train ENAS.')
    parser.add_argument('--controller_train_steps',    type=int,     default='50',   help='.')
    parser.add_argument('--controller_num_aggregate',  type=int,     default='20',   help='.')
    parser.add_argument('--controller_entropy_weight', type=float,   default='0.0001',   help='The weight for the entropy of the controller.')
    parser.add_argument('--controller_bl_dec'        , type=float,   default='0.99',   help='.')
    parser.add_argument('--controller_num_samples'   , type=int,     default='100',   help='.')
    # log
    parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',           type=str,   default='/home/city/disk/log/enas',    help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset',   type=str,   default='/home/city/.torch/NAS-Bench-102-v1_0-e61699.pth',    help='The path to load the architecture dataset (nas-benchmark).')
    parser.add_argument('--print_freq',         type=int,   default='200',    help='print frequency (default: 200)')
    parser.add_argument('--rand_seed',          type=int,   default='-1',    help='manual seed')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    main(args)


