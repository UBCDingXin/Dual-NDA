print("\n===================================================================================================")

import os
import argparse
import shutil
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
import csv
from tqdm import tqdm
import gc
import h5py

### import my stuffs ###
from opts import cnn_opts
from models import *
from utils import *
from train_cnn import train_cnn, test_cnn





#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = cnn_opts()
print(args)

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# CNN settings
## lr decay scheme
lr_decay_epochs = (args.lr_decay_epochs).split("_")
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]

#-------------------------------
# output folders

output_directory = os.path.join(args.root_path, 'output/CNN')
cnn_info = '{}_lr_{}_decay_{}'.format(args.cnn_name, args.lr_base, args.weight_decay)
    
os.makedirs(output_directory, exist_ok=True)

#-------------------------------
# some functions
min_label_before_shift = args.min_label
max_label_after_shift = args.max_label+np.abs(min_label_before_shift)

def fn_norm_labels(labels):
    '''
    labels: unnormalized labels
    '''
    labels += np.abs(min_label_before_shift)
    labels /= max_label_after_shift
    assert labels.min()>=0 and labels.max()<=1.0
    return labels

def fn_denorm_labels(labels):
    '''
    labels: normalized labels
    '''
    # assert labels.min()>=0 and labels.max()<=1.0
    labels = labels*max_label_after_shift - np.abs(min_label_before_shift)

    if isinstance(labels, np.ndarray):
        return labels.astype(int)
    elif torch.is_tensor(labels):
        return labels.type(torch.int)
    else:
        return int(labels)





#######################################################################################
'''                                Data loader                                      '''
#######################################################################################
print('\n Loading real data...')
data_filename = args.data_path + '/SteeringAngle_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels_train = hf['labels'][:]
labels_train = labels_train.astype(float)
images_train = hf['images'][:]
hf.close()

# remove too small angles and too large angles
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels_train>q1)*(labels_train<q2)==True)[0]
labels_train = labels_train[indx]
images_train = images_train[indx]
assert len(labels_train)==len(images_train)

## unique labels
unique_labels = np.sort(np.array(list(set(labels_train))))

# ## for each age, take no more than args.max_num_img_per_label images
# image_num_threshold = args.max_num_img_per_label
# print("\n Original training set has {} images; For each age, take no more than {} images>>>".format(len(images_train), image_num_threshold))

# sel_indx = []
# for i in tqdm(range(len(unique_labels))):
#     indx_i = np.where(labels_train == unique_labels[i])[0]
#     if len(indx_i)>image_num_threshold:
#         np.random.shuffle(indx_i)
#         indx_i = indx_i[0:image_num_threshold]
#     sel_indx.append(indx_i)
# sel_indx = np.concatenate(sel_indx, axis=0)
# images_train = images_train[sel_indx]
# labels_train = labels_train[sel_indx]
# print("\r {} training images left.".format(len(images_train)))

## normalize to [0,1]
labels_train = fn_norm_labels(labels_train)

## number of real images
nreal = len(labels_train)
assert len(labels_train) == len(images_train)

## data loader for the training set and test set
trainset = IMGs_dataset(images_train, labels_train, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)



#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################

### model initialization

net = cnn_dict[args.cnn_name]()
net = nn.DataParallel(net)

### start training
filename_ckpt = os.path.join(output_directory, 'ckpt_{}_epoch_{}_last.pth'.format(args.cnn_name, args.epochs))
print('\n' + filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Start training the {} >>>".format(args.cnn_name))

    path_to_ckpt_in_train = output_directory + '/ckpts_in_train/{}'.format(cnn_info)    
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    train_cnn(net=net, net_name=args.cnn_name, train_images=images_train, train_labels=labels_train, testloader=trainloader, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=args.save_freq, batch_size=args.batch_size_train, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, path_to_ckpt = path_to_ckpt_in_train, fn_denorm_labels=fn_denorm_labels)

    # store model
    torch.save({
        'net_state_dict': net.state_dict(),
    }, filename_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.cnn_name))
    checkpoint = torch.load(filename_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])
#end if

print("\n===================================================================================================")
