print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm, trange
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image
import sys

### import my stuffs ###
from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)
from utils import IMGs_dataset, SimpleProgressBar, compute_entropy, predict_class_labels
from models import *
from train_ccgan import train_ccgan, sample_ccgan_given_labels
from train_net_for_label_embed import train_net_embed, train_net_y2h
from eval_metrics import cal_FID, cal_labelscore, inception_score


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)     

#-------------------------------
# Embedding
base_lr_x2y = 0.01
base_lr_y2h = 0.01


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
data_filename = args.data_path + '/SteeringAngle_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(float)
images = hf['images'][:]
hf.close()

# remove too small angles and too large angles
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels>q1)*(labels<q2)==True)[0]
labels = labels[indx]
images = images[indx]
assert len(labels)==len(images)

raw_images = copy.deepcopy(images) #backup images;
raw_labels = copy.deepcopy(labels) #backup raw labels; we may normalize labels later


#-------------------------------
# some functions

# min_label_before_shift = args.min_label
# max_label_after_shift = args.max_label+np.abs(min_label_before_shift)
min_label_before_shift = np.min(labels)
max_label_after_shift = np.max(labels+np.abs(min_label_before_shift))

def fn_norm_labels(labels):
    '''
    labels: unnormalized labels
    '''
    output = labels + np.abs(min_label_before_shift)
    output = output/max_label_after_shift
    assert output.min()>=0 and output.max()<=1.0
    return output

def fn_denorm_labels(labels):
    '''
    labels: normalized labels
    '''
    # assert labels.min()>=0 and labels.max()<=1.0
    output = labels*max_label_after_shift - np.abs(min_label_before_shift)

    if isinstance(output, np.ndarray):
        return output.astype(float)
    elif torch.is_tensor(output):
        return output.type(torch.float)
    else:
        return float(output)

# for each angle, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images = images[sel_indx]
labels = labels[sel_indx]
print("{} images left and there are {} unique labels".format(len(images), len(set(labels))))

## print number of images for each label
unique_labels_tmp = np.sort(np.array(list(set(labels))))
num_img_per_label_all = np.zeros(len(unique_labels_tmp))
for i in range(len(unique_labels_tmp)):
    indx_i = np.where(labels==unique_labels_tmp[i])[0]
    num_img_per_label_all[i] = len(indx_i)
#print(list(num_img_per_label_all))
data_csv = np.concatenate((unique_labels_tmp.reshape(-1,1), num_img_per_label_all.reshape(-1,1)), 1)
np.savetxt(wd + '/label_dist.csv', data_csv, delimiter=',')


## replicate minority samples to alleviate the imbalance issue
max_num_img_per_label_after_replica = args.max_num_img_per_label_after_replica
if max_num_img_per_label_after_replica>1:
    unique_labels_replica = np.sort(np.array(list(set(labels))))
    num_labels_replicated = 0
    print("Start replicating minority samples >>>")
    for i in tqdm(range(len(unique_labels_replica))):
        curr_label = unique_labels_replica[i]
        indx_i = np.where(labels == curr_label)[0]
        if len(indx_i) < max_num_img_per_label_after_replica:
            num_img_less = max_num_img_per_label_after_replica - len(indx_i)
            indx_replica = np.random.choice(indx_i, size = num_img_less, replace=True)
            if num_labels_replicated == 0:
                images_replica = images[indx_replica]
                labels_replica = labels[indx_replica]
            else:
                images_replica = np.concatenate((images_replica, images[indx_replica]), axis=0)
                labels_replica = np.concatenate((labels_replica, labels[indx_replica]))
            num_labels_replicated+=1
    #end for i
    images = np.concatenate((images, images_replica), axis=0)
    labels = np.concatenate((labels, labels_replica))
    print("We replicate {} images and labels \n".format(len(images_replica)))
    del images_replica, labels_replica; gc.collect()


# normalize labels
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
labels = fn_norm_labels(labels)
assert labels.min()>=0 and labels.max()<=1.0
print("\r Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))
unique_labels_norm = np.sort(np.array(list(set(labels))))
print("\r There are {} unique labels.".format(len(unique_labels_norm)))

if args.kernel_sigma<0:
    std_label = np.std(labels)
    args.kernel_sigma = 1.06*std_label*(len(labels))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels), std_label, args.kernel_sigma))

if args.kappa<0:
    n_unique = len(unique_labels_norm)

    diff_list = []
    for i in range(1,n_unique):
        diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
    kappa_base = np.abs(args.kappa)*np.max(np.array(diff_list))

    if args.threshold_type=="hard":
        args.kappa = kappa_base
    else:
        args.kappa = 1/kappa_base**2




#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(wd, 'output/{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}'.format(args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa, args.loss_type_gan, args.num_D_steps, args.num_grad_acc_d, args.num_grad_acc_g, args.batch_size_disc, args.batch_size_gene))
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)

path_to_embed_models = os.path.join(wd, 'output/embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)

if args.mae_filter or args.niqe_filter:
    fake_data_folder = os.path.join(path_to_output, 'bad_fake_data')
    os.makedirs(fake_data_folder, exist_ok=True)
    save_setting_folder = os.path.join(fake_data_folder, "{}".format(args.setting_name))
    os.makedirs(save_setting_folder, exist_ok=True)



#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################

net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_{}.pth'.format(args.net_embed, args.epoch_cnn_embed, args.seed))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_net_y2h_epoch_{}_seed_{}.pth'.format(args.epoch_net_y2h, args.seed))

print("\n "+net_embed_filename_ckpt)
print("\n "+net_y2h_filename_ckpt)

labels_train_embed = raw_labels + np.abs(min_label_before_shift)
labels_train_embed /= max_label_after_shift
unique_labels_norm_embed = np.sort(np.array(list(set(labels_train_embed))))
print("\n labels_train_embed: min={}, max={}".format(np.min(labels_train_embed), np.max(labels_train_embed)))
trainset_embed = IMGs_dataset(raw_images, labels_train_embed, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset_embed, batch_size=args.batch_size_embed, shuffle=True, num_workers=args.num_workers) #use data before replication

if args.net_embed == "ResNet18_embed":
    net_embed = ResNet18_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet34_embed":
    net_embed = ResNet34_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet50_embed":
    net_embed = ResNet50_embed(dim_embed=args.dim_embed)
net_embed = net_embed.cuda()
net_embed = nn.DataParallel(net_embed)

net_y2h = model_y2h(dim_embed=args.dim_embed)
net_y2h = net_y2h.cuda()
net_y2h = nn.DataParallel(net_y2h)

## (1). Train net_embed first: x2h+h2y
if not os.path.isfile(net_embed_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")
    net_embed = train_net_embed(net=net_embed, net_name=args.net_embed, trainloader=trainloader_embed_net, testloader=None, epochs=args.epoch_cnn_embed, resume_epoch = args.resumeepoch_cnn_embed, lr_base=base_lr_x2y, lr_decay_factor=0.1, lr_decay_epochs=[80, 140], weight_decay=1e-4, path_to_ckpt = path_to_embed_models)
    # save model
    torch.save({
    'net_state_dict': net_embed.state_dict(),
    }, net_embed_filename_ckpt)
else:
    print("\n net_embed ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_filename_ckpt)
    net_embed.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

## (2). Train y2h
#train a net which maps a label back to the embedding space
if not os.path.isfile(net_y2h_filename_ckpt):
    print("\n Start training net_y2h >>>")
    net_y2h = train_net_y2h(unique_labels_norm, net_y2h, net_embed, epochs=args.epoch_net_y2h, lr_base=base_lr_y2h, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128)
    # save model
    torch.save({
    'net_state_dict': net_y2h.state_dict(),
    }, net_y2h_filename_ckpt)
else:
    print("\n net_y2h ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_y2h_filename_ckpt)
    net_y2h.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

##some simple test
indx_tmp = np.arange(len(unique_labels_norm_embed))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_norm_embed[indx_tmp].reshape(-1,1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).cuda()
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1,1).type(torch.float).cuda()
labels_noise_tmp = torch.clamp(labels_tmp+epsilons_tmp, 0.0, 1.0)
net_embed.eval()
net_h2y = net_embed.module.h2y
net_y2h.eval()
with torch.no_grad():
    labels_hidden_tmp = net_y2h(labels_tmp)
    labels_noise_hidden_tmp = net_y2h(labels_noise_tmp)
    labels_rec_tmp = net_h2y(labels_hidden_tmp).cpu().numpy().reshape(-1,1)
    labels_noise_rec_tmp = net_h2y(labels_noise_hidden_tmp).cpu().numpy().reshape(-1,1)
    labels_hidden_tmp = labels_hidden_tmp.cpu().numpy()
    labels_noise_hidden_tmp = labels_noise_hidden_tmp.cpu().numpy()
labels_tmp = labels_tmp.cpu().numpy()
labels_noise_tmp = labels_noise_tmp.cpu().numpy()
results1 = np.concatenate((labels_tmp, labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results1)

labels_diff = (labels_tmp-labels_noise_tmp)**2
hidden_diff = np.mean((labels_hidden_tmp-labels_noise_hidden_tmp)**2, axis=1, keepdims=True)
results2 = np.concatenate((labels_diff, hidden_diff), axis=1)
print("\n labels diff vs hidden diff")
print(results2)

#put models on cpu
net_embed = net_embed.cpu()
net_h2y = net_h2y.cpu()
del net_embed, net_h2y; gc.collect()
net_y2h = net_y2h.cpu()


#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("CcGAN: {}, {}, Sigma is {}, Kappa is {}.".format(args.GAN_arch, args.threshold_type, args.kernel_sigma, args.kappa))
save_images_in_train_folder = save_images_folder + '/images_in_train'
os.makedirs(save_images_in_train_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training:")
#----------------------------------------------
# cGAN: treated as a classification dataset
Filename_GAN = save_models_folder + '/ckpt_niter_{}.pth'.format(args.niters_gan)
print('\r', Filename_GAN)
if not os.path.isfile(Filename_GAN):
    netG = CcGAN_SAGAN_Generator(dim_z=args.dim_gan, dim_embed=args.dim_embed)
    netD = CcGAN_SAGAN_Discriminator(dim_embed=args.dim_embed)
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    # Start training
    netG, netD = train_ccgan(args.kernel_sigma, args.kappa, images, labels, netG, netD, net_y2h, save_images_folder=save_images_in_train_folder, save_models_folder = save_models_folder)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(Filename_GAN)
    netG = CcGAN_SAGAN_Generator(dim_z=args.dim_gan, dim_embed=args.dim_embed).cuda()
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])

def fn_sampleGAN_given_labels(labels, batch_size, to_numpy=True, denorm=True, verbose=True):
    fake_images, fake_labels = sample_ccgan_given_labels(netG, net_y2h, labels, batch_size = batch_size, to_numpy=to_numpy, denorm=denorm, verbose=verbose)
    return fake_images, fake_labels

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))


#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID:
    # for FID
    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = args.eval_ckpt_path + '/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class_eval(num_classes=5, ngpu = torch.cuda.device_count()).cuda() # give scenes
    Filename_PreCNNForEvalGANs_Diversity = args.eval_ckpt_path + '/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_20_seed_2020_classify_5_scenes_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = args.eval_ckpt_path + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])


    #####################
    # generate nfake images
    print("Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))
    eval_labels = np.linspace(np.min(raw_labels), np.max(raw_labels), args.num_eval_labels) #not normalized
    eval_labels_norm = fn_norm_labels(eval_labels)
    # eval_labels_norm = (eval_labels + np.abs(min_label_before_shift)) / max_label_after_shift

    for i in range(args.num_eval_labels):
        curr_label = eval_labels_norm[i]
        if i == 0:
            fake_labels_assigned = np.ones(args.nfake_per_label)*curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(args.nfake_per_label)*curr_label))
    
    start_time = timeit.default_timer()
    fake_images, _ = fn_sampleGAN_given_labels(fake_labels_assigned, args.samp_batch_size)
    print("End sampling! We got {} fake images. Time elapse: {:.3f}.".format(len(fake_images), timeit.default_timer()-start_time))
    assert len(fake_images) == args.nfake_per_label*args.num_eval_labels
    assert len(fake_labels_assigned) == args.nfake_per_label*args.num_eval_labels

    ## dump fake images for computing NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        if args.niqe_dump_path=="None":
            dump_fake_images_folder = save_images_folder + '/fake_images'
        else:
            dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
        # dump_fake_images_folder = save_images_folder + '/fake_images_for_NIQE_nfake_{}'.format(len(fake_images))
        os.makedirs(dump_fake_images_folder, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            # label_i = fake_labels_assigned[i]*max_label_after_shift-np.abs(min_label_before_shift)
            label_i = fn_denorm_labels(fake_labels_assigned[i])
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i].astype(np.uint8)
            # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
        # sys.exit()

    print("End sampling! We got {} fake images.".format(len(fake_images)))


    #####################
    # prepare real/fake images and labels
    # real_images = (raw_images/255.0-0.5)/0.5
    real_images = raw_images
    real_labels = fn_norm_labels(raw_labels)
    # real_labels = (raw_labels + np.abs(min_label_before_shift)) / max_label_after_shift
    nfake_all = len(fake_images)
    nreal_all = len(real_images)
    
    
    if args.comp_IS_and_FID_only:
        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 200, resize = None, norm_img = True)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        #####################
        # IS: Evaluate IS on all fake images
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=5, net=PreNetDiversity, cuda=True, batch_size=200, splits=10, normalize_img = True)
        print("\n IS of {} fake images: {}({}).".format(nfake_all, IS, IS_std))
    
    
    else:
        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (not normalized range, i.e., [min_label,max_label]). The center of the sliding window locate on [min_label+R,...,max_label-R].
        center_start = np.min(raw_labels)+args.FID_radius
        center_stop = np.max(raw_labels)-args.FID_radius
        centers_loc = np.linspace(center_start, center_stop, args.FID_num_centers) #not normalized

        # output center locations for computing NIQE
        filename_centers = wd + '/steering_angle_centers_loc_for_NIQE.txt'
        np.savetxt(filename_centers, centers_loc)

        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = fn_norm_labels(center - args.FID_radius)
            interval_stop = fn_norm_labels(center + args.FID_radius)
            indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            real_images_curr = (real_images_curr/255.0-0.5)/0.5
            num_realimgs_over_centers[i] = len(real_images_curr)
            indx_fake = np.where((fake_labels_assigned>=interval_start)*(fake_labels_assigned<=interval_stop)==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_images_curr = (fake_images_curr/255.0-0.5)/0.5
            fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
            ## FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=200, resize = None)
            ## Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=200, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            ## Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=min_label_before_shift, max_label_after_shift=max_label_after_shift, batch_size=200, resize = None, num_workers=args.num_workers)
            ## print
            print("\n [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
        # end for i
        # average over all centers
        print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n Entropy over centers: {}({}); min/max: {}/{}.".format(np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(path_to_output, 'fid_ls_entropy_over_centers')
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=200, resize = None, norm_img = True)
        print("\n {}: FID of {} fake images: {}.".format(args.GAN_arch, nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=min_label_before_shift, max_label_after_shift=max_label_after_shift, batch_size=200, resize = None, norm_img = True, num_workers=args.num_workers)
        print("\n {}: overall LS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, ls_mean_overall, ls_std_overall))


        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(path_to_output, 'eval_results_{}.txt'.format(args.GAN_arch))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Radius: {}; # Centers: {}.  \n".format(args.FID_radius, args.FID_num_centers))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            # eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {:.3f} ({:.3f}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))


#######################################################################################
'''                        Generate bad fake samples                              '''
#######################################################################################

# target_labels = np.linspace(np.min(raw_labels), np.max(raw_labels), args.num_eval_labels) #not normalized
# target_labels_norm = fn_norm_labels(target_labels)
# assert target_labels_norm.min()>=0 and target_labels_norm.max()<=1
# num_target_labels = len(target_labels_norm)

target_labels_norm = copy.deepcopy(unique_labels_norm)
assert target_labels_norm.min()>=0 and target_labels_norm.max()<=1
num_target_labels = len(target_labels_norm)


####################################################################
''' Generate bad fake images based on MAE filtering '''
if args.mae_filter:
    print("\n-------------------------------------------")
    print("\r Generate bad fake images based on MAE...")

    ## load pre-trained cnn
    filter_precnn_net = cnn_dict[args.samp_filter_precnn_net]().cuda()
    filter_precnn_net = nn.DataParallel(filter_precnn_net)
    filter_precnn_ckpt_path = os.path.join(wd, 'output/CNN/ckpt_{}_epoch_200_last.pth'.format(args.samp_filter_precnn_net))
    checkpoint = torch.load(filter_precnn_ckpt_path)
    filter_precnn_net.load_state_dict(checkpoint['net_state_dict'])

    ### initial generation for computing the filtering threshold
    print("\r Initial generation for deciding the filtering threshold...")

    def compute_mae(net, input_images, input_labels, batch_size=args.samp_batch_size, verbose=False):
        ''' 
        input_images: tensor, unnormalized
        input_labels: tensor, normalized
        '''
        assert input_images.max()>1.0
        input_images = ((input_images/255.0)-0.5)/0.5
        assert input_images.min()>=-1 and input_images.max()<=1.0

        net = net.cuda()
        net.eval()

        assert len(input_images)==len(input_labels)
        n = len(input_images)

        input_images = torch.cat((input_images,input_images[0:batch_size]),dim=0)
        input_labels = torch.cat((input_labels,input_labels[0:batch_size]),dim=0)

        if verbose:
            pb = SimpleProgressBar()
        mae_all = []
        indx_ = 0
        with torch.no_grad():
            while indx_*batch_size<n:
                batch_images = input_images[(indx_*batch_size):((indx_+1)*batch_size)].float().cuda()
                batch_labels_pred = net(batch_images)
                batch_labels = input_labels[(indx_*batch_size):((indx_+1)*batch_size)].view(-1).cpu()
                batch_labels_pred = batch_labels_pred.view(-1).cpu()
                batch_mae = torch.abs(batch_labels-batch_labels_pred)
                mae_all.append(batch_mae.numpy())
                indx_+=1
                if verbose:
                    pb.update(min(float(indx_*batch_size)/n, 1)*100)
        mae_all = np.concatenate(mae_all, axis=0)
        mae_all = mae_all[0:n]

        return mae_all

    # filter_precnn_net.eval()
    # mae_all = []
    # for i in trange(num_target_labels):
    #     label_i = target_labels_norm[i]
    #     images_i, labels_i = fn_sampleGAN_given_labels(label_i*np.ones([args.samp_filter_nfake_per_label_burnin,1]), args.samp_batch_size, to_numpy=False, denorm=True, verbose=False)
    #     assert labels_i.min()>=0 and labels_i.max()<=1
    #     assert images_i.max()>1.0
    #     mae_i = compute_mae(net=filter_precnn_net, input_images=images_i, input_labels=labels_i, batch_size=args.samp_batch_size)
    #     mae_all.append(mae_i)
    # ## for i
    # mae_all = np.array(mae_all)
    # mae_cutoff_point_overall = np.quantile(mae_all, q=args.samp_filter_mae_percentile_threshold)

    # ## plot of MAEs
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.plot(fn_denorm_labels(target_labels_norm), fn_denorm_labels(mae_all), color='tab:red', linestyle='-', marker='o', alpha=0.85)
    # ax.set_xlabel("MAEs", fontsize=12)
    # ax.set_ylabel("Target Labels", fontsize=12)
    # fig.savefig(os.path.join(save_setting_folder, 'plot_of_fake_data_MAE_Filter_{}.pdf'.format(args.samp_filter_mae_percentile_threshold)), bbox_inches='tight')


    ### compute the overall cutoff point
    for i in range(num_target_labels):
        curr_label = target_labels_norm[i]
        if i == 0:
            fake_labels_assigned = np.ones(args.samp_filter_nfake_per_label_burnin)*curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(args.samp_filter_nfake_per_label_burnin)*curr_label))
    fake_images, fake_labels = fn_sampleGAN_given_labels(fake_labels_assigned, args.samp_batch_size, to_numpy=False, denorm=True, verbose=True)
    n_burnin = len(fake_images)

    # fake_images = torch.cat((fake_images, fake_images[0:args.samp_batch_size]), dim=0)
    # fake_labels = torch.cat((fake_labels, fake_labels[0:args.samp_batch_size]), dim=0)
    
    print("\r Compute MAEs of fake images...")
    pb = SimpleProgressBar()
    filter_precnn_net.eval()
    assert fake_labels.min()>=0 and fake_labels.max()<=1
    mae_all = compute_mae(net=filter_precnn_net, input_images=fake_images, input_labels=fake_labels, batch_size=args.samp_batch_size)
    mae_cutoff_point_overall = np.quantile(mae_all, q=args.samp_filter_mae_percentile_threshold)
    print("\r Cutoff point is {}.".format(mae_cutoff_point_overall))


    # mae_all = []
    # n_got=0
    # indx=0
    # while n_got<n_burnin:
    #     batch_images = fake_images[(indx*args.samp_batch_size):((indx+1)*args.samp_batch_size)]
    #     batch_labels = fake_labels[(indx*args.samp_batch_size):((indx+1)*args.samp_batch_size)]
    #     assert batch_labels.min()>=0 and batch_labels.max()<=1
    #     assert batch_images.max()>1.0
    #     batch_mae = compute_mae(net=filter_precnn_net, input_images=batch_images, input_labels=batch_labels, batch_size=args.samp_batch_size)
    #     mae_all.append(batch_mae)
    #     n_got+=args.samp_batch_size
    #     indx+=1
    #     pb.update(min(float(n_got)/n_burnin, 1)*100)
    # ##end while
    # mae_all = np.concatenate(mae_all, aixs=0)
    # mae_all = mae_all[0:n_burnin]
    # fake_images = fake_images[0:n_burnin]
    # fake_labels = fake_labels[0:n_burnin]
    # mae_cutoff_point_overall = np.quantile(mae_all, q=args.samp_filter_mae_percentile_threshold)



    ### final generation
    print('\r Final generation for data output...')
    with torch.no_grad():
        fake_images = []
        fake_labels = []
        fake_labels_pred = []
                
        for i in tqdm(range(num_target_labels)):
            label_i = target_labels_norm[i]

            fake_images_i = []
            fake_labels_i = []
            fake_labels_pred_i = []

            ngot_ = 0
            while ngot_<args.samp_filter_nfake_per_label:
                batch_images, batch_labels = fn_sampleGAN_given_labels(label_i*np.ones([args.samp_batch_size,1]), args.samp_batch_size, to_numpy=False, denorm=True, verbose=False)
                assert batch_labels.min()>=0 and batch_labels.max()<=1
                assert batch_images.max()>1.0

                batch_images = ((batch_images/255.0)-0.5)/0.5
                batch_images = batch_images.float().cuda()
                pred_labels = filter_precnn_net(batch_images)
                
                batch_labels = batch_labels.view(-1).cpu()
                pred_labels = pred_labels.view(-1).cpu()
                
                batch_loss = (torch.abs(batch_labels-pred_labels)).numpy()

                indx_sel = np.where(batch_loss>=mae_cutoff_point_overall)[0]

                if len(indx_sel)>0:
                    batch_images = batch_images.cpu().numpy()
                    batch_images = batch_images[indx_sel]
                    batch_labels = batch_labels.cpu().numpy()
                    batch_labels = batch_labels[indx_sel]

                    assert batch_images.min()>=-1 and batch_images.max()<=1
                    assert batch_labels.min()>= 0 and batch_labels.max()<=1
                    batch_images = ((batch_images*0.5)+0.5)*255.0
                    batch_labels = fn_denorm_labels(batch_labels)

                    fake_images_i.append(batch_images)
                    fake_labels_i.append(batch_labels)
                    fake_labels_pred_i.append(pred_labels.numpy())

                    ngot_+=len(batch_images)
            ##end while
            fake_images_i = np.concatenate(fake_images_i, axis=0)
            fake_labels_i = np.concatenate(fake_labels_i, axis=0)
            fake_labels_pred_i = np.concatenate(fake_labels_pred_i, axis=0)

            fake_images.append(fake_images_i[0:args.samp_filter_nfake_per_label])
            fake_labels.append(fake_labels_i[0:args.samp_filter_nfake_per_label])
            fake_labels_pred.append(fake_labels_pred_i[0:args.samp_filter_nfake_per_label])
        ## for i
        fake_images = np.concatenate(fake_images, axis=0)
        fake_labels = np.concatenate(fake_labels, axis=0)
        fake_labels_pred = np.concatenate(fake_labels_pred, axis=0)
        fake_labels_pred = fn_denorm_labels(fake_labels_pred)



    ## dump fake samples into h5 file
    assert fake_images.max()>1.0
    assert fake_labels.max()>1.0
    print("\r fake images' range: MIN={}, MAX={}".format(np.min(fake_images), np.max(fake_images)))
    dump_fake_images_filename = os.path.join(save_setting_folder, 'badfake_MAE{}_nfake{}.h5'.format(args.samp_filter_mae_percentile_threshold, len(fake_images)))
    print(dump_fake_images_filename)
    with h5py.File(dump_fake_images_filename, "w") as f:
        f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
        f.create_dataset('fake_labels', data = fake_labels, dtype='float')   
    print("\n The dim of the fake dataset based on filtering: ", fake_images.shape)
    print("\n The range of generated fake dataset: MIN={}, MAX={}.".format(fake_labels.min(), fake_labels.max()))

    # ## draw same example bad fake images
    # n_row = 20
    # n_col = 10
    # filename_fake_images = os.path.join(save_setting_folder, 'badfake_imgs_filter{}_{}x{}.png'.format(args.samp_filter_mae_percentile_threshold, n_row, n_col))
    # start_label = np.quantile(fake_labels, 0.05)
    # end_label = np.quantile(fake_labels, 0.95)
    # selected_labels = np.linspace(start_label, end_label, num=n_row)
    # indx_show = []
    # for i in range(len(selected_labels)):
    #     label_i = selected_labels[i]
    #     indx_i = np.where(np.abs(fake_labels-label_i)<1e-6)[0]
    #     np.random.shuffle(indx_i)
    #     indx_show.append(indx_i[0:n_col])
    # ##end for i
    # indx_show = np.concatenate(indx_show, axis=0)
    # example_fake_images = fake_images[indx_show]
    # example_fake_images = (example_fake_images/255.0-0.5)/0.5
    # example_fake_images = torch.from_numpy(example_fake_images)
    # save_image(example_fake_images.data, filename_fake_images, nrow=n_col, normalize=True)  

    
    


####################################################################
''' Generate bad fake images based on NIQE filtering '''
if args.niqe_filter:
    
    print("\n-------------------------------------------")
    print("\r Generate fake images for niqe filtering...")
    fake_images = []
    fake_labels = []
    for i in trange(len(target_labels_norm)):
        fake_images_i, fake_labels_i = fn_sampleGAN_given_labels(target_labels_norm[i]*np.ones([args.niqe_nfake_per_label_burnin,1]), args.samp_batch_size, to_numpy=True, denorm=True, verbose=False)
        ### denormalize labels
        assert fake_labels_i.min()>=0 and fake_labels_i.max()<=1
        fake_labels_i = fn_denorm_labels(fake_labels_i)
        ### append
        fake_images.append(fake_images_i)
        fake_labels.append(fake_labels_i)
    ##end for i
    fake_images = np.concatenate(fake_images, axis=0)
    fake_labels = np.concatenate(fake_labels, axis=0)
    assert fake_images.max()>1
    assert fake_labels.max()>1


    print("\n Dumping bad fake images for NIQE...")
    if args.niqe_dump_path=="None":
        dump_fake_images_folder = save_setting_folder + '/fake_images_for_NIQE'
    else:
        dump_fake_images_folder = args.niqe_dump_path + '/fake_images_for_NIQE'
    os.makedirs(dump_fake_images_folder, exist_ok=True)
    for i in tqdm(range(len(fake_images))):
        filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, float(fake_labels[i]))
        os.makedirs(os.path.dirname(filename_i), exist_ok=True)
        image_i = fake_images[i]
        # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
        image_i = np.uint8(image_i)
        image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
        image_i_pil.save(filename_i)
    #end for i
    # sys.exit()




print("\n===================================================================================================")