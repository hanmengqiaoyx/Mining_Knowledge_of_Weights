'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress. Just for Linux System
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim

import matplotlib.pyplot as plt
from matplotlib import gridspec


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


term_width = 100                                            # if Windows
# _, term_width = os.popen('stty size', 'r').read().split()   # if unix
# term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def save_loss(filename, epoch, loss, auc=0):
    if not os.path.isdir('out_record'):
        os.mkdir('out_record')
    with open('./out_record/loss_record_' + filename + '.admo', 'a') as f:
        f.write(str(epoch) + ' ' + str(loss) + '\n')


def draw_sample(samples, filename, dirpath='fig_out/'):
    samples = samples.data.cpu().numpy()[:4]
    # fig = plt.figure(figsize=(10, 10))
    # gs = gridspec.GridSpec(4, 4)
    # gs.update(wspace=0.05, hspace=0.05)

    fig, ax = plt.subplots(1, 4, figsize=(25, 25))

    for x in range(4):
        ax[x].imshow(samples[x].squeeze(), cmap=plt.cm.gray)

    # for i, sample in enumerate(samples):
    #     ax = plt.subplot(gs[i])
    #     plt.axis('off')
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_aspect('equal')
    #
    #     # pdb.set_trace()
    #
    #     plt.imshow(sample[0], cmap='Greys_r')

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def draw_npy(samples, filenames, dirpath='fig_out/', append=''):
    
    import numpy as np

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    sp = samples.data.cpu().numpy()
    for i, img in enumerate(sp):
        np.save(os.path.join(dirpath, filenames[i].replace('/', '_') + append + '.npy'), img[0])
