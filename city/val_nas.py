import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time, time_string_short
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_102_api  import NASBench102API as API
from torchvision import transforms, datasets


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)


search_model = torch.load('/home/city/disk/log/builds-darts/darts2_0024_20200107_112_71.52.pth')
# search_model.apply(weights_init)
print(nn.functional.softmax(search_model.arch_parameters, dim = -1))
print(search_model.genotype())
# for w in search_model.get_weights():
#     if len(w.size())== 1:
#         nn.init.constant(w,0)
#     else:
#         nn.init.xavier_uniform(w)
#

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[128, 128, 128], std=[50, 50, 50])
])
val_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[128, 128, 128], std=[50, 50, 50])
])

train_data = datasets.ImageFolder(root='/home/city/Projects/build_assessment/data/train',
                                  transform=train_transform)
valid_data = datasets.ImageFolder(root='/home/city/Projects/build_assessment/data/val',
                                  transform=val_transform)
print(len(train_data))

train_split = []
valid_split = []

for i in range(len(train_data)):
    if i % 2 == 0:
        train_split.append(i)
    else:
        valid_split.append(i)
search_data = SearchDataset('builds', train_data, train_split, valid_split)

search_loader = torch.utils.data.DataLoader(search_data,
                                            batch_size=32, shuffle=True,
                                            num_workers=4, pin_memory=True
                                            )
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=32, shuffle=True,
                                           num_workers=2, pin_memory=True
                                           )

# w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
optim = torch.optim.Adadelta(search_model.get_weights())
criterion = torch.nn.CrossEntropyLoss()

base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
time_start = time.time()
time_pre = time.time()




search_model.eval()
# search_model.eval()
for step, (base_inputs, base_targets) in enumerate(valid_loader):
    base_targets = base_targets.cuda(non_blocking=True)
    # print('in',base_inputs[0])

    # optim.zero_grad()
    with torch.no_grad():
        _, logits = search_model(base_inputs.cuda())

        arch_prec1 = obtain_accuracy(logits.data, base_targets.data)
    arch_top1.update(arch_prec1[0])

print('val_acc %.2f used %.2fs' % (arch_top1.avg,time.time()-time_pre))
time_pre=time.time()

reg_lambda = 0.001
for epoch in range(10000):
    search_model.train()
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(search_loader):
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # print(torch.mean(base_targets.float()))

        optim.zero_grad()
        _, logits = search_model(base_inputs.cuda())
        base_loss = criterion(logits, base_targets)


        l2_reg = 0
        for W in search_model.get_weights():
            l2_reg += W.norm(2)
        base_loss += reg_lambda * l2_reg




        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(search_model.parameters(), 5)
        optim.step()
        base_prec1 = obtain_accuracy(logits.data, base_targets.data)
        base_top1.update(base_prec1[0])

        optim.zero_grad()
        _, logits = search_model(arch_inputs.cuda())
        base_loss = criterion(logits, arch_targets)
        # print(base_loss)
        l2_reg = 0
        for W in search_model.get_weights():
            l2_reg += torch.norm(W)
            # print(l2_reg)
        base_loss += reg_lambda * l2_reg
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(search_model.parameters(), 5)
        optim.step()
        base_prec1 = obtain_accuracy(logits.data, arch_targets.data)
        # print(base_prec1)
        base_top1.update(base_prec1[0])
    search_model.eval()
    for step, (base_inputs, base_targets) in enumerate(valid_loader):
        base_targets = base_targets.cuda(non_blocking=True)

        # optim.zero_grad()
        with torch.no_grad():
            _, logits = search_model(base_inputs.cuda())

            arch_prec1 = obtain_accuracy(logits.data, base_targets.data)
        arch_top1.update(arch_prec1[0])

    print('Epoch %04d acc %.2f val_acc %.2f used %.2fs' % (epoch+1, base_top1.avg, arch_top1.avg,time.time()-time_pre))
    time_pre=time.time()