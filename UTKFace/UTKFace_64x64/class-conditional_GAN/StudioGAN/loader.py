# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/loader.py

from os.path import dirname, abspath, exists, join
import sys
import glob
import json
import os
import random
import warnings
import numpy as np
from tqdm import tqdm,trange
from PIL import Image
import timeit

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import wandb

from data_util import Dataset_
from utils.style_ops import grid_sample_gradfix
from utils.style_ops import conv2d_gradfix
from metrics.inception_net import InceptionV3
from sync_batchnorm.batchnorm import convert_model
from worker import WORKER
import utils.log as log
import utils.losses as losses
import utils.ckpt as ckpt
import utils.misc as misc
import utils.custom_ops as custom_ops
import models.model as model
import metrics.preparation as pp

### my evaluation
from extra_models import *
from eval_metrics import cal_FID, cal_labelscore, inception_score, compute_entropy, predict_class_labels


def load_worker(local_rank, cfgs, gpus_per_node, run_name, hdf5_path):
    # -----------------------------------------------------------------------------
    # define default variables for loading ckpt or evaluating the trained GAN model.
    # -----------------------------------------------------------------------------
    load_train_dataset = cfgs.RUN.train + cfgs.RUN.do_eval + cfgs.RUN.GAN_train + cfgs.RUN.GAN_test ## my modified load_train_dataset
    len_eval_metrics = 0 if cfgs.RUN.eval_metrics == ["none"] else len(cfgs.RUN.eval_metrics)
    load_eval_dataset = cfgs.RUN.do_eval + len_eval_metrics + cfgs.RUN.save_real_images + cfgs.RUN.k_nearest_neighbor + \
        cfgs.RUN.frequency_analysis + cfgs.RUN.tsne_analysis + cfgs.RUN.intra_class_fid ## my modified load_eval_dataset
    train_sampler, eval_sampler = None, None
    step, epoch, topk, best_step, best_fid, best_ckpt_path, lecam_emas, is_best = \
        0, 0, cfgs.OPTIMIZATION.batch_size, 0, None, None, None, False
    mu, sigma, real_feats, eval_model, num_rows, num_cols = None, None, None, None, 10, 8
    aa_p = cfgs.AUG.ada_initial_augment_p
    if cfgs.AUG.ada_initial_augment_p != "N/A":
        aa_p = cfgs.AUG.ada_initial_augment_p
    else:
        aa_p = cfgs.AUG.apa_initial_augment_p

    loss_list_dict = {"gen_loss": [], "dis_loss": [], "cls_loss": []}
    num_eval = {}
    metric_dict_during_train = {}
    if "none" in cfgs.RUN.eval_metrics:
        cfgs.RUN.eval_metrics = []
    if "is" in cfgs.RUN.eval_metrics:
        metric_dict_during_train.update({"IS": [], "Top1_acc": [], "Top5_acc": []})
    if "fid" in cfgs.RUN.eval_metrics:
        metric_dict_during_train.update({"FID": []})
    if "prdc" in cfgs.RUN.eval_metrics:
        metric_dict_during_train.update({"Improved_Precision": [], "Improved_Recall": [], "Density":[], "Coverage": []})

    # -----------------------------------------------------------------------------
    # determine cuda, cudnn, and backends settings.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.fix_seed:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False

    if cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]:
        # Improves training speed
        conv2d_gradfix.enabled = True
        # Avoids errors with the augmentation pipe
        grid_sample_gradfix.enabled = True
        if cfgs.RUN.mixed_precision:
            # Allow PyTorch to internally use tf32 for matmul
            torch.backends.cuda.matmul.allow_tf32 = False
            # Allow PyTorch to internally use tf32 for convolutions
            torch.backends.cudnn.allow_tf32 = False

    # -----------------------------------------------------------------------------
    # initialize all processes and fix seed of each process
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        global_rank = cfgs.RUN.current_node * (gpus_per_node) + local_rank
        print("Use GPU: {global_rank} for training.".format(global_rank=global_rank))
        misc.setup(global_rank, cfgs.OPTIMIZATION.world_size, cfgs.RUN.backend)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    misc.fix_seed(cfgs.RUN.seed + global_rank)

    # -----------------------------------------------------------------------------
    # Intialize python logger.
    # -----------------------------------------------------------------------------
    if local_rank == 0:
        logger = log.make_logger(cfgs.RUN.save_dir, run_name, None)
        if cfgs.RUN.ckpt_dir is not None and cfgs.RUN.freezeD == -1:
            folder_hier = cfgs.RUN.ckpt_dir.split("/")
            if folder_hier[-1] == "":
                folder_hier.pop()
            logger.info("Run name : {run_name}".format(run_name=folder_hier.pop()))
        else:
            logger.info("Run name : {run_name}".format(run_name=run_name))
        for k, v in cfgs.super_cfgs.items():
            logger.info("cfgs." + k + " =")
            logger.info(json.dumps(vars(v), indent=2))
    else:
        logger = None


    # -----------------------------------------------------------------------------
    # load train and evaluation datasets.
    # -----------------------------------------------------------------------------
    if load_train_dataset:
        if local_rank == 0:
            logger.info("Load {name} train dataset for training.".format(name=cfgs.DATA.name))
        train_dataset = Dataset_(data_name=cfgs.DATA.name,
                                data_dir=cfgs.RUN.data_dir,
                                train=True,
                                img_size=cfgs.DATA.img_size,
                                num_classes=cfgs.DATA.num_classes,
                                label_lb=cfgs.DATA.min_label,
                                label_ub=cfgs.DATA.max_label,
                                crop_long_edge=cfgs.PRE.crop_long_edge,
                                resize_size=cfgs.PRE.resize_size,
                                resizer=None if hdf5_path is not None else cfgs.RUN.pre_resizer,
                                random_flip=cfgs.PRE.apply_rflip,
                                normalize=True,
                                hdf5_path=hdf5_path,
                                load_data_in_memory=cfgs.RUN.load_data_in_memory)

        min_label_before_shift, max_label_after_shift = train_dataset._return_min_max()
        
        if local_rank == 0:
            logger.info("Train dataset size: {dataset_size}".format(dataset_size=len(train_dataset)))
    else:
        train_dataset = None

    if load_eval_dataset:
        if local_rank == 0:
            logger.info("Load {name} {ref} dataset for evaluation.".format(name=cfgs.DATA.name, ref=cfgs.RUN.ref_dataset))
        eval_dataset = Dataset_(data_name=cfgs.DATA.name,
                                data_dir=cfgs.RUN.data_dir,
                                train= False,
                                img_size=cfgs.DATA.img_size,
                                num_classes=cfgs.DATA.num_classes,
                                label_lb=cfgs.DATA.min_label,
                                label_ub=cfgs.DATA.max_label,
                                crop_long_edge=False if cfgs.DATA.name in cfgs.MISC.no_proc_data else True,
                                resize_size=None if cfgs.DATA.name in cfgs.MISC.no_proc_data else cfgs.DATA.img_size,
                                resizer=cfgs.RUN.pre_resizer,
                                random_flip=False,
                                hdf5_path=None,
                                normalize=True,
                                load_data_in_memory=False)
        
        min_label_before_shift, max_label_after_shift = eval_dataset._return_min_max()

        if local_rank == 0:
            logger.info("Eval dataset size: {dataset_size}".format(dataset_size=len(eval_dataset)))
    else:
        eval_dataset = None
    

    # -----------------------------------------------------------------------------
    # define a distributed sampler for DDP train and evaluation.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        cfgs.OPTIMIZATION.batch_size = cfgs.OPTIMIZATION.batch_size//cfgs.OPTIMIZATION.world_size
        if cfgs.RUN.train:
            train_sampler = DistributedSampler(train_dataset,
                                            num_replicas=cfgs.OPTIMIZATION.world_size,
                                            rank=local_rank,
                                            shuffle=True,
                                            drop_last=True)
            topk = cfgs.OPTIMIZATION.batch_size

        if cfgs.RUN.do_eval:
            eval_sampler = DistributedSampler(eval_dataset,
                                            num_replicas=cfgs.OPTIMIZATION.world_size,
                                            rank=local_rank,
                                            shuffle=False,
                                            drop_last=False)

    cfgs.OPTIMIZATION.basket_size = cfgs.OPTIMIZATION.batch_size*\
                                    cfgs.OPTIMIZATION.acml_steps*\
                                    cfgs.OPTIMIZATION.d_updates_per_step

    # -----------------------------------------------------------------------------
    # define dataloaders for train and evaluation.
    # -----------------------------------------------------------------------------
    if load_train_dataset:
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=cfgs.OPTIMIZATION.basket_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=True,
                                      num_workers=cfgs.RUN.num_workers,
                                      sampler=train_sampler,
                                      drop_last=True,
                                      persistent_workers=True)
    else:
        train_dataloader = None

    if load_eval_dataset:
        eval_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=cfgs.OPTIMIZATION.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=cfgs.RUN.num_workers,
                                     sampler=eval_sampler,
                                     drop_last=False)
    else:
        eval_dataloader = None

    # -----------------------------------------------------------------------------
    # load a generator and a discriminator
    # if cfgs.MODEL.apply_g_ema is True, load an exponential moving average generator (Gen_ema).
    # -----------------------------------------------------------------------------
    Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis, ema =\
        model.load_generator_discriminator(DATA=cfgs.DATA,
                                           OPTIMIZATION=cfgs.OPTIMIZATION,
                                           MODEL=cfgs.MODEL,
                                           STYLEGAN=cfgs.STYLEGAN,
                                           MODULES=cfgs.MODULES,
                                           RUN=cfgs.RUN,
                                           device=local_rank,
                                           logger=logger)

    if local_rank != 0:
        custom_ops.verbosity = "none"

    # -----------------------------------------------------------------------------
    # define optimizers for adversarial training
    # -----------------------------------------------------------------------------
    cfgs.define_optimizer(Gen, Dis)

    # -----------------------------------------------------------------------------
    # load the generator and the discriminator from a checkpoint if possible
    # -----------------------------------------------------------------------------
    if cfgs.RUN.ckpt_dir is not None:
        # if local_rank == 0:
        #     os.remove(join(cfgs.RUN.save_dir, "logs", run_name + ".log"))
        run_name, step, epoch, topk, aa_p, best_step, best_fid, best_ckpt_path, lecam_emas, logger =\
            ckpt.load_StudioGAN_ckpts(ckpt_dir=cfgs.RUN.ckpt_dir,
                                      load_best=cfgs.RUN.load_best,
                                      Gen=Gen,
                                      Dis=Dis,
                                      g_optimizer=cfgs.OPTIMIZATION.g_optimizer,
                                      d_optimizer=cfgs.OPTIMIZATION.d_optimizer,
                                      run_name=run_name,
                                      apply_g_ema=cfgs.MODEL.apply_g_ema,
                                      Gen_ema=Gen_ema,
                                      ema=ema,
                                      is_train=cfgs.RUN.train,
                                      RUN=cfgs.RUN,
                                      logger=logger,
                                      global_rank=global_rank,
                                      device=local_rank,
                                      cfg_file=cfgs.RUN.cfg_file)

        if topk == "initialize":
            topk == cfgs.OPTIMIZATION.batch_size
        if cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]:
            ema.ema_rampup = "N/A" # disable EMA rampup
            if cfgs.MODEL.backbone == "stylegan3" and cfgs.STYLEGAN.stylegan3_cfg == "stylegan3-r":
                cfgs.STYLEGAN.blur_init_sigma = "N/A" # disable blur rampup
        if cfgs.AUG.apply_ada:
            cfgs.AUG.ada_kimg = 100 # make ADA react faster at the beginning

    if cfgs.RUN.ckpt_dir is None or cfgs.RUN.freezeD != -1:
        if local_rank == 0:
            cfgs.RUN.ckpt_dir = ckpt.make_ckpt_dir(join(cfgs.RUN.save_dir, "checkpoints", run_name))
        dict_dir = join(cfgs.RUN.save_dir, "statistics", run_name)
        loss_list_dict = misc.load_log_dicts(directory=dict_dir, file_name="losses.npy", ph=loss_list_dict)
        metric_dict_during_train = misc.load_log_dicts(directory=dict_dir, file_name="metrics.npy", ph=metric_dict_during_train)

    if not cfgs.RUN.train:
        if cfgs.RUN.path_to_G != "None":
            print("\n Loading pre-trained Generator!")
            checkpoint = torch.load(cfgs.RUN.path_to_G)
            Gen.load_state_dict(checkpoint['state_dict'])
        if cfgs.RUN.path_to_G_ema != "None":
            print("\n Loading pre-trained Generator with EMA!")
            checkpoint = torch.load(cfgs.RUN.path_to_G_ema)
            Gen_ema.load_state_dict(checkpoint['state_dict'])

    # -----------------------------------------------------------------------------
    # prepare parallel training
    # -----------------------------------------------------------------------------
    if cfgs.OPTIMIZATION.world_size > 1:
        Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis =\
        model.prepare_parallel_training(Gen=Gen,
                                        Gen_mapping=Gen_mapping,
                                        Gen_synthesis=Gen_synthesis,
                                        Dis=Dis,
                                        Gen_ema=Gen_ema,
                                        Gen_ema_mapping=Gen_ema_mapping,
                                        Gen_ema_synthesis=Gen_ema_synthesis,
                                        MODEL=cfgs.MODEL,
                                        world_size=cfgs.OPTIMIZATION.world_size,
                                        distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                        synchronized_bn=cfgs.RUN.synchronized_bn,
                                        apply_g_ema=cfgs.MODEL.apply_g_ema,
                                        device=local_rank)

    # -----------------------------------------------------------------------------
    # load a pre-trained network (InceptionV3, SwAV, DINO, or Swin-T)
    # -----------------------------------------------------------------------------
    if cfgs.DATA.name in ["ImageNet", "Baby_ImageNet", "Papa_ImageNet", "Grandpa_ImageNet"]:
        num_eval = {"train": 50000, "valid": len(eval_dataloader.dataset)}
    else:
        if eval_dataloader is not None:
            num_eval[cfgs.RUN.ref_dataset] = len(eval_dataloader.dataset)
        else:
            num_eval["train"], num_eval["valid"], num_eval["test"] = 50000, 50000, 50000

    if len(cfgs.RUN.eval_metrics) or cfgs.RUN.intra_class_fid:
        eval_model = pp.LoadEvalModel(eval_backbone=cfgs.RUN.eval_backbone,
                                      post_resizer=cfgs.RUN.post_resizer,
                                      world_size=cfgs.OPTIMIZATION.world_size,
                                      distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                      device=local_rank)

    if "fid" in cfgs.RUN.eval_metrics:
        mu, sigma = pp.prepare_moments(data_loader=eval_dataloader,
                                       eval_model=eval_model,
                                       quantize=True,
                                       cfgs=cfgs,
                                       logger=logger,
                                       device=local_rank)

    if "prdc" in cfgs.RUN.eval_metrics:
        if cfgs.RUN.distributed_data_parallel:
            prdc_sampler = DistributedSampler(eval_dataset,
                                              num_replicas=cfgs.OPTIMIZATION.world_size,
                                              rank=local_rank,
                                              shuffle=True,
                                              drop_last=False)
        else:
            prdc_sampler = None

        prdc_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=cfgs.OPTIMIZATION.batch_size,
                                     shuffle=(prdc_sampler is None),
                                     pin_memory=True,
                                     num_workers=cfgs.RUN.num_workers,
                                     sampler=prdc_sampler,
                                     drop_last=False)

        real_feats = pp.prepare_real_feats(data_loader=prdc_dataloader,
                                           eval_model=eval_model,
                                           num_feats=num_eval[cfgs.RUN.ref_dataset],
                                           quantize=True,
                                           cfgs=cfgs,
                                           logger=logger,
                                           device=local_rank)

    if cfgs.RUN.calc_is_ref_dataset:
        pp.calculate_ins(data_loader=eval_dataloader,
                         eval_model=eval_model,
                         quantize=True,
                         splits=1,
                         cfgs=cfgs,
                         logger=logger,
                         device=local_rank)

    # -----------------------------------------------------------------------------
    # initialize WORKER for training and evaluating GAN
    # -----------------------------------------------------------------------------
    worker = WORKER(
        cfgs=cfgs,
        run_name=run_name,
        Gen=Gen,
        Gen_mapping=Gen_mapping,
        Gen_synthesis=Gen_synthesis,
        Dis=Dis,
        Gen_ema=Gen_ema,
        Gen_ema_mapping=Gen_ema_mapping,
        Gen_ema_synthesis=Gen_ema_synthesis,
        ema=ema,
        eval_model=eval_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        global_rank=global_rank,
        local_rank=local_rank,
        mu=mu,
        sigma=sigma,
        real_feats=real_feats,
        logger=logger,
        aa_p=aa_p,
        best_step=best_step,
        best_fid=best_fid,
        best_ckpt_path=best_ckpt_path,
        lecam_emas=lecam_emas,
        num_eval=num_eval,
        loss_list_dict=loss_list_dict,
        metric_dict_during_train=metric_dict_during_train,
    )

    # -----------------------------------------------------------------------------
    # train GAN until "total_steps" generator updates
    # -----------------------------------------------------------------------------
    if cfgs.RUN.train:
        if global_rank == 0:
            logger.info("Start training!")

        worker.training, worker.topk = True, topk
        worker.prepare_train_iter(epoch_counter=epoch)
        while step <= cfgs.OPTIMIZATION.total_steps:
            # print(step)
            if cfgs.OPTIMIZATION.d_first:
                real_cond_loss, dis_acml_loss = worker.train_discriminator(current_step=step)
                gen_acml_loss = worker.train_generator(current_step=step)
            else:
                gen_acml_loss = worker.train_generator(current_step=step)
                real_cond_loss, dis_acml_loss = worker.train_discriminator(current_step=step)

            if global_rank == 0 and (step + 1) % cfgs.RUN.print_freq == 0:
                worker.log_train_statistics(current_step=step,
                                            real_cond_loss=real_cond_loss,
                                            gen_acml_loss=gen_acml_loss,
                                            dis_acml_loss=dis_acml_loss)
            step += 1

            if cfgs.LOSS.apply_topk:
                if (epoch + 1) == worker.epoch_counter:
                    epoch += 1
                    worker.topk = losses.adjust_k(current_k=worker.topk,
                                                  topk_gamma=cfgs.LOSS.topk_gamma,
                                                  inf_k=int(cfgs.OPTIMIZATION.batch_size * cfgs.LOSS.topk_nu))

            if step % cfgs.RUN.save_freq == 0:
                # visuailize fake images
                if global_rank == 0:
                   worker.visualize_fake_images(num_cols=num_cols, current_step=step)

                # evaluate GAN for monitoring purpose
                if len(cfgs.RUN.eval_metrics) :
                    is_best = worker.evaluate(step=step, metrics=cfgs.RUN.eval_metrics, writing=True, training=True)

                # save GAN in "./checkpoints/RUN_NAME/*"
                if global_rank == 0:
                    worker.save(step=step, is_best=is_best)

                # stop processes until all processes arrive
                if cfgs.RUN.distributed_data_parallel:
                    dist.barrier(worker.group)

        if global_rank == 0:
            logger.info("End of training!")

    # -----------------------------------------------------------------------------
    # re-evaluate the best GAN and conduct ordered analyses
    # -----------------------------------------------------------------------------
    worker.training, worker.epoch_counter = False, epoch
    worker.gen_ctlr.standing_statistics = cfgs.RUN.standing_statistics
    worker.gen_ctlr.standing_max_batch = cfgs.RUN.standing_max_batch
    worker.gen_ctlr.standing_step = cfgs.RUN.standing_step

    # if global_rank == 0:
    #     best_step = ckpt.load_best_model(ckpt_dir=cfgs.RUN.ckpt_dir,
    #                                      Gen=Gen,
    #                                      Dis=Dis,
    #                                      apply_g_ema=cfgs.MODEL.apply_g_ema,
    #                                      Gen_ema=Gen_ema,
    #                                      ema=ema)
        

    # if len(cfgs.RUN.eval_metrics):
    #     for e in range(cfgs.RUN.num_eval):
    #         if global_rank == 0:
    #             print(""), logger.info("-" * 80)
    #         _ = worker.evaluate(step=best_step, metrics=cfgs.RUN.eval_metrics, writing=False, training=False)

    # if cfgs.RUN.save_real_images:
    #     if global_rank == 0: print(""), logger.info("-" * 80)
    #     worker.save_real_images()

    # if cfgs.RUN.save_fake_images:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.save_fake_images(num_images=cfgs.RUN.save_fake_images_num)

    # if cfgs.RUN.vis_fake_images:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.visualize_fake_images(num_cols=num_cols, current_step=best_step)

    # if cfgs.RUN.k_nearest_neighbor:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.run_k_nearest_neighbor(dataset=eval_dataset, num_rows=num_rows, num_cols=num_cols)

    # if cfgs.RUN.interpolation:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.run_linear_interpolation(num_rows=num_rows, num_cols=num_cols, fix_z=True, fix_y=False)
    #     worker.run_linear_interpolation(num_rows=num_rows, num_cols=num_cols, fix_z=False, fix_y=True)

    # if cfgs.RUN.frequency_analysis:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.run_frequency_analysis(dataloader=eval_dataloader)

    # if cfgs.RUN.tsne_analysis:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.run_tsne(dataloader=eval_dataloader)

    # if cfgs.RUN.intra_class_fid:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.calculate_intra_class_fid(dataset=eval_dataset)

    # if cfgs.RUN.semantic_factorization:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.run_semantic_factorization(num_rows=cfgs.RUN.num_semantic_axis,
    #                                       num_cols=num_cols,
    #                                       maximum_variations=cfgs.RUN.maximum_variations)
    # if cfgs.RUN.GAN_train:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.compute_GAN_train_or_test_classifier_accuracy_score(GAN_train=True, GAN_test=False)

    # if cfgs.RUN.GAN_test:
    #     if global_rank == 0:
    #         print(""), logger.info("-" * 80)
    #     worker.compute_GAN_train_or_test_classifier_accuracy_score(GAN_train=False, GAN_test=True)




    # -----------------------------------------------------------------------------
    # generate fake images for evaluation
    # -----------------------------------------------------------------------------
    if cfgs.RUN.do_eval:
        
        # if global_rank == 0:
        #     eval_folder = os.path.join(cfgs.RUN.save_dir, 'evaluation')
        #     os.makedirs(eval_folder, exist_ok=True)
        eval_folder = os.path.join(cfgs.RUN.save_dir, 'evaluation')
        os.makedirs(eval_folder, exist_ok=True)

        ########################
        # load eval models
        #for FID
        PreNetFID = encoder(dim_bottleneck=512).cuda()
        PreNetFID = torch.nn.DataParallel(PreNetFID)
        checkpoint_PreNet = torch.load(cfgs.RUN.eval_ckpt_path_FID)
        PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

        # Diversity: entropy of predicted races within each eval center
        PreNetDiversity = ResNet34_class_eval(num_classes=5, ngpu = torch.cuda.device_count()).cuda() #5 races
        checkpoint_PreNet = torch.load(cfgs.RUN.eval_ckpt_path_Div)
        PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

        # for LS
        PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
        checkpoint_PreNet = torch.load(cfgs.RUN.eval_ckpt_path_LS)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

        #####################
        # generate nfake images
        print("\n Start sampling {} fake images per label from GAN >>>".format(cfgs.RUN.nfake_per_label))
        labels_all = eval_dataset.labels
        class_cutoff_points = train_dataset.class_cutoff_points
        assert len(class_cutoff_points)-1 == cfgs.DATA.num_classes
        eval_labels = np.sort(np.array(list(set(labels_all)))) #not normalized #evaluating at training labels
        print("\n There are {} distinct eval labels.".format(len(eval_labels)))

        ## determine conditional labels
        for i in range(len(eval_labels)):
            curr_label = eval_labels[i]
            if i == 0:
                fake_labels_assigned = np.ones(cfgs.RUN.nfake_per_label)*curr_label
            else:
                fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(cfgs.RUN.nfake_per_label)*curr_label))
        nfake = len(fake_labels_assigned)
        print("\n We will generate {} fake images.".format(len(fake_labels_assigned)))

        ## convert regression labels into class labels
        given_class_labels = np.zeros(nfake)
        for i in range(nfake):
            curr_given_label = fake_labels_assigned[i]
            diff_tmp = class_cutoff_points - curr_given_label
            indx_nonneg = np.where(diff_tmp>=0)[0]
            if len(indx_nonneg)==1: #the last element of diff_tmp is non-negative
                curr_given_class_label = cfgs.DATA.num_classes-1
                assert indx_nonneg[0] == cfgs.DATA.num_classes
            elif len(indx_nonneg)>1:
                if diff_tmp[indx_nonneg[0]]>0:
                    curr_given_class_label = indx_nonneg[0] - 1
                else:
                    curr_given_class_label = indx_nonneg[0]
            given_class_labels[i] = int(curr_given_class_label)
        assert given_class_labels.min()==0 and given_class_labels.max()==cfgs.DATA.num_classes-1

        ## generate fake images given labels
        start = timeit.default_timer()
        fake_images = []
        for i in trange(cfgs.DATA.num_classes):
            nfake_i = len(np.where(given_class_labels==i)[0])
            fake_images_i = worker.generate_fake_images(label=i, nfake_per_label=nfake_i)
            assert len(fake_images_i)==nfake_i
            fake_images.append(fake_images_i)
        ##end for
        fake_images = np.concatenate(fake_images, axis=0)
        assert len(fake_images) == nfake
        stop = timeit.default_timer()
        sampling_time = stop-start
        print("\r Sampling completes; Time elapses: {}s".format(sampling_time))
        
        ## dump fake images for evaluation: NIQE
        if cfgs.RUN.dump_fake_for_NIQE:                
            if cfgs.RUN.dump_fake_img_path=="None":
                dump_fake_images_folder = os.path.join(eval_folder, 'fake_images')
            else:
                dump_fake_images_folder = os.path.join(cfgs.RUN.dump_fake_img_path, 'fake_images')
            os.makedirs(dump_fake_images_folder, exist_ok=True)

            for i in tqdm(range(len(fake_images))):
                label_i = fake_labels_assigned[i]
                filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, int(label_i))
                os.makedirs(os.path.dirname(filename_i), exist_ok=True)
                image_i = fake_images[i]
                image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
                image_i_pil.save(filename_i)
            #end for i


        #####################
        # normalize real images
        real_images = (eval_dataset.data/255.0-0.5)/0.5
        real_labels = labels_all/cfgs.DATA.max_label

        fake_images = (fake_images/255.0-0.5)/0.5
        fake_labels_assigned = fake_labels_assigned/cfgs.DATA.max_label

        nfake_all = len(fake_images)
        nreal_all = len(real_images)

        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (not normalized range, i.e., [min_label,max_label]). The center of the sliding window locate on [min_label+R,...,max_label-R].
        center_start = np.min(labels_all)+cfgs.RUN.FID_radius
        center_stop = np.max(labels_all)-cfgs.RUN.FID_radius

        # centers_loc = np.arange(center_start, center_stop+1) #not normalized, for UTKFace only

        if cfgs.RUN.FID_num_centers<=0 and cfgs.RUN.FID_radius==0: #completely overlap
            centers_loc = eval_labels #not normalized
        elif cfgs.RUN.FID_num_centers>0:
            centers_loc = np.linspace(center_start, center_stop, cfgs.RUN.FID_num_centers) #not normalized
        else:
            print("\n Error.")

        print("\n ")
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = (center - cfgs.RUN.FID_radius)
            interval_stop = (center + cfgs.RUN.FID_radius)
            indx_real = np.where((real_labels>=interval_start/cfgs.DATA.max_label)*(real_labels<=interval_stop/cfgs.DATA.max_label)==True)[0]
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            num_realimgs_over_centers[i] = len(real_images_curr)
            indx_fake = np.where((fake_labels_assigned>=(interval_start/cfgs.DATA.max_label))*(fake_labels_assigned<=(interval_stop/cfgs.DATA.max_label))==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
            # FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=cfgs.RUN.eval_batch_size, resize = None)
            # Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=cfgs.RUN.eval_batch_size)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=cfgs.DATA.max_label, batch_size = cfgs.RUN.eval_batch_size, resize = None)

            print("\r [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
        # end for i
        # average over all centers
        print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n Entropy over centers: {}({}); min/max: {}/{}.".format(np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(eval_folder, "fid_ls_entropy_over_centers")
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)


        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=0, max_label_after_shift=cfgs.DATA.max_label, batch_size = cfgs.RUN.eval_batch_size, resize = None)
        print("\n Overall LS of {} fake images: {}({}).".format(nfake_all, ls_mean_overall, ls_std_overall))


        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = cfgs.RUN.eval_batch_size, resize = None)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        #####################
        # IS: Evaluate IS on all fake images
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=5, net=PreNetDiversity, cuda=True, batch_size=cfgs.RUN.eval_batch_size, splits=10, normalize_img=False)
        print("\n IS of {} fake images: {}({}).".format(nfake_all, IS, IS_std))

        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(eval_folder, 'eval_results.txt')
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Eval Mode: Radius: {}; # Centers: {}; Sampling time: {:.3f}.  \n".format(cfgs.RUN.FID_radius, cfgs.RUN.FID_num_centers, sampling_time))
            print(cfgs, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers)))
            eval_results_logging_file.write("\n Diversity: {:.3f} ({:.3f}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n IS: {:.3f} ({:.3f}).".format(IS, IS_std))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))





    ####################################################################
    # final
    if global_rank == 0:
        wandb.finish()
