"""Main training script. Currently only supports the BBCPose dataset
"""
from apex.parallel import DistributedDataParallel as DDP
from utils.visualizer import dump_image, project_heatmaps_colorized
from models.losses import Vgg19PerceptualLoss, GANLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataloaders.bbc_pose_dataset import BBCPoseDataset
from torch import optim
from models.discriminator import MultiscaleDiscriminator
from utils.utils import initialize_distributed, parse_all_args,\
     get_learning_rate, log_iter, save_options, reduce_tensor, \
     get_model, load_weights, save_weights
import torch
import numpy as np
import torch.nn.functional
import torch.nn as nn
import os
import argparse
torch.backends.cudnn.benchmark = True


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.1)


def setup_dataloaders(config):
    """Setup dataloaders for respective datasets

    Args:
        config (dict): dictionary of runtime configuration options

    Returns:
        train_dataloader (torch.utils.data.Dataloader): Dataloader for training split.
        val_dataloader (torch.utils.data.Dataloader): Dataloader for validation split.
        train_sampler (torch.utils.data.distributed.DistributedSampler): DDP sampler if using DDP.
    """

    # setup the dataset
    if config['dataset'] == 'bbc_pose':
        train_dataset = BBCPoseDataset(config, 'train')
        val_dataset = BBCPoseDataset(config, 'validation')
    else:
        print("No such dataset!")
        exit(-1)
    # distributed sampler if world size > 1
    train_sampler = None
    val_sampler = None
    if config['use_DDP']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # wrap the datasets in a dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=(train_sampler is None),
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  sampler=train_sampler)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=config['num_workers'],
                                pin_memory=True,
                                sampler=val_sampler)

    return train_dataloader, val_dataloader, train_sampler


def run_val(model, criterion, val_dataloader, vis_path=None):
    """Validation pass
    Runs in no_grad with model in eval mode
    sets model back to train() at the end
    """
    model.eval()
    num_batches = len(val_dataloader)
    mean_loss = 0
    cnt = 0
    with torch.no_grad():
        for curr_batch in val_dataloader:
            print("Val iter %d / %d" % (cnt, len(val_dataloader)))
            input_a = curr_batch['input_a'].cuda()
            input_b = curr_batch['input_b'].cuda()
            target = curr_batch['target'].cuda()
            imnames = curr_batch['imname']
            output_dict = model(input_a, input_b)
            output_recon = output_dict['reconstruction']
            loss = criterion(output_dict['reconstruction'], target)
            mean_loss = mean_loss + loss/num_batches
            if vis_path is not None and cnt == 0:
                try:
                    os.mkdir(vis_path)
                except OSError:
                    print('Folder exists')

                # dump validation images into vis_path folder
                B, C, H, W = input_a.shape
                visualization_centers = output_dict['vis_centers']

                for b in range(B):
                    imname, _ = imnames[b].split('.')
                    _ = dump_image(target[b].cpu(), None, os.path.join(vis_path, imname+'.png'))
                    _ = dump_image(output_recon[b].cpu(), None, os.path.join(vis_path, imname+'o.png'))

            cnt = cnt + 1
    model.train()
    return mean_loss


def run_visualization(output_dict, output_recon, target, input_a, input_b, out_path, tb_logger, step, warped_heatmap=None):
    """Function for preparing visualizations in the tensorboard log
    """
    visualization_centers = output_dict['vis_centers']
    x = visualization_centers[0]
    y = visualization_centers[1]
    x_b, y_b = output_dict['input_b_gauss_params'][0], output_dict['input_b_gauss_params'][1]

    vis1img = dump_image(target[0].cpu(), (x[0], y[0]), os.path.join(out_path, 'vis1.png'))
    vis1oimg = dump_image(output_recon[0], (x[0], y[0]), os.path.join(out_path, 'vis1o.png'))
    if 'background_recon' in output_dict.keys():
        vis1baimg = dump_image(output_dict['background_recon'][0], None, os.path.join(out_path, 'vis1ba.png'))
        vis1dimg = dump_image(output_dict['decoded_foreground'][0], None, os.path.join(out_path, 'vis1d.png'))

        target_imgs = np.concatenate((vis1img, vis1oimg, vis1baimg, vis1dimg), axis=1)
        tb_logger.add_image("target_reconstruction_background_foreground", target_imgs, global_step=step, dataformats='HWC')

        mask_imgs = torch.cat((output_dict['input_a_fg_mask'][0], output_dict['input_b_fg_mask'][0]), dim=2)
        tb_logger.add_image("inputamask_inputbmask", mask_imgs, global_step=step, dataformats='CHW')
    else:
        target_and_recon = np.concatenate((vis1img, vis1oimg), axis=1)
        tb_logger.add_image("target_reconstruction", target_and_recon, global_step=step, dataformats='HWC')
    vis1aimg = dump_image(input_a[0].cpu(), (x[0], y[0]), os.path.join(out_path, 'vis1a.png'))
    vis1bimg = dump_image(input_b[0].cpu(), (x_b[0], y_b[0]), os.path.join(out_path, 'vis1b.png'))
    input_imgs = np.concatenate((vis1aimg, vis1bimg), axis=1)

    tb_logger.add_image("input_a_b", input_imgs, global_step=step, dataformats='HWC')
    if 'weighted_center_prediction' in output_dict.keys():
        object_center = output_dict['weighted_center_prediction'][0]
        object_center_x = object_center[0]
        object_center_y = object_center[1]
        predicted_centers = output_dict['fg_predicted_centers']
        vis1imgcenter = plot_offsets(target[0], predicted_centers[0], os.path.join(out_path, 'vis1c.png'))
        tb_logger.add_image("predicted_center", vis1imgcenter, global_step = step, dataformats='HWC')

    heat_maps = output_dict['input_a_heatmaps'][0].data.cpu().numpy()
    heat_maps_out = project_heatmaps_colorized(heat_maps)
    tb_logger.add_image("raw_heatmap", heat_maps_out.astype(np.uint8), global_step=step, dataformats='CHW')

    if warped_heatmap is not None:
        warped_heatmap = warped_heatmap.data.cpu().numpy()
        warped_heatmaps_out = project_heatmaps_colorized(warped_heatmap)
        tb_logger.add_image("warped_raw_heatmap", warped_heatmap_out.astype(np.uint8), global_step=step, dataformats='CHW')


def apply_GAN_criterion(output_recon, target, predicted_keypoints,
                        discriminator, criterionGAN):
    """Sub-routine for applying adversarial loss within the main train loop
    Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/pix2pixHD_model.py, which in turn was adpated from 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
    Args:
        output_recon (torch.tensor): reconstruction from decoder.
        target (torch.tensor): reference image.
        predicted_keypoints (torch.tensor): predicted gauss maps.
        discriminator (torch.nn.Module): discriminator model.
        criterionGAN (torch.nn.Module): decoder criterion.
    Returns:
        Loss values for the generator and discriminator
    """
    pred_fake_D = discriminator(output_recon.detach(), predicted_keypoints)
    loss_D_fake = criterionGAN(pred_fake_D, False)
    pred_real = discriminator(target.detach(), predicted_keypoints)
    loss_D_real = criterionGAN(pred_real, True)

    pred_fake = discriminator(output_recon, predicted_keypoints)
    loss_G_GAN = criterionGAN(pred_fake, True)

    return loss_G_GAN, loss_D_real, loss_D_fake


def main(config):
    save_path = config['save_path']
    epochs = config['epochs']
    os.environ['TORCH_HOME'] = config['torch_home']
    distributed =  config['use_DDP']
    start_ep = 0
    start_cnt = 0

    # initialize model
    print("Initializing model...")
    if distributed:
        initialize_distributed(config)
    rank = config['rank']

    # map string name to class constructor
    model = get_model(config)
    model.apply(init_weights)
    if config['resume_ckpt'] is not None:
        # load weights from checkpoint
        state_dict = load_weights(config['resume_ckpt'])
        model.load_state_dict(state_dict)

    print("Moving model to GPU")
    model.cuda(torch.cuda.current_device())
    print("Setting up losses")

    if config['use_vgg']:
        criterionVGG = Vgg19PerceptualLoss(config['reduced_w'])
        criterionVGG.cuda()
        validationLoss = criterionVGG
    if config['use_gan']:
        use_sigmoid = config['no_lsgan']
        disc_input_channels = 3
        discriminator = MultiscaleDiscriminator(disc_input_channels, config['ndf'], config['n_layers_D'], 'instance',
                                                use_sigmoid, config['num_D'], False, False)
        discriminator.apply(init_weights)
        if config['resume_ckpt_D'] is not None:
            # load weights from checkpoint
            print("Resuming discriminator from %s" %(config['resume_ckpt_D']))
            state_dict = load_weights(config['resume_ckpt_D'])
            discriminator.load_state_dict(state_dict)

        discriminator.cuda(torch.cuda.current_device())
        criterionGAN = GANLoss(use_lsgan=not config['no_lsgan'])
        criterionGAN.cuda()
        criterionFeat = nn.L1Loss().cuda()

    # initialize dataloader
    print("Setting up dataloaders...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config)
    print("Done!")
    # run the training loop
    print("Initializing optimizers...")
    optimizer_G = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if config['resume_ckpt_opt_G'] is not None:
        optimizer_G_state_dict = torch.load(config['resume_ckpt_opt_G'], map_location=lambda storage, loc: storage)
        optimizer_G.load_state_dict(optimizer_G_state_dict)
    if config['use_gan']:
        optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'])
        if config['resume_ckpt_opt_D'] is not None:
            optimizer_D_state_dict = torch.load(config['resume_ckpt_opt_D'], map_location=lambda storage, loc: storage)
            optimizer_D.load_state_dict(optimizer_D_state_dict)

    print("Done!")

    if distributed:
        print("Moving model to DDP...")
        model = DDP(model)
        if config['use_gan']:
            discriminator = DDP(discriminator, delay_allreduce=True)
        print("Done!")

    tb_logger = None
    if rank == 0:
        tb_logdir = os.path.join(save_path, 'tbdir')
        if not os.path.exists(tb_logdir):
            os.makedirs(tb_logdir)
        tb_logger = SummaryWriter(tb_logdir)
            # run training
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        log_name = os.path.join(save_path, 'loss_log.txt')
        opt_name = os.path.join(save_path, 'opt.yaml')
        print(config)
        save_options(opt_name, config)
        log_handle = open(log_name, 'a')

    print("Starting training")
    cnt = start_cnt
    assert(config['use_warped'] or config['use_temporal'])

    for ep in range(start_ep, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        for curr_batch in train_dataloader:
            optimizer_G.zero_grad()
            input_a = curr_batch['input_a'].cuda()
            target = curr_batch['target'].cuda()
            if config['use_warped'] and config['use_temporal']:
                input_a = torch.cat((input_a, input_a), 0)
                input_b = torch.cat((curr_batch['input_b'].cuda(), curr_batch['input_temporal'].cuda()), 0)
                target = torch.cat((target, target), 0)
            elif config['use_temporal']:
                input_b = curr_batch['input_temporal'].cuda()
            elif config['use_warped']:
                input_b = curr_batch['input_b'].cuda()

            output_dict = model(input_a, input_b)
            output_recon = output_dict['reconstruction']

            loss_vgg = loss_G_GAN = loss_G_feat = 0
            if config['use_vgg']:
                loss_vgg = criterionVGG(output_recon, target) * config['vgg_lambda']
            if config['use_gan']:
                predicted_landmarks = output_dict['input_a_gauss_maps']
                # output_dict['reconstruction'] can be considered normalized
                loss_G_GAN, loss_D_real, loss_D_fake = apply_GAN_criterion(output_recon, target, predicted_landmarks.detach(),
                                                                           discriminator, criterionGAN)
                loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_G = loss_G_GAN + loss_G_feat + loss_vgg
            loss_G.backward()
            # grad_norm clipping
            if not config['no_grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_G.step()
            if config['use_gan']:
                optimizer_D.zero_grad()
                loss_D.backward()
                # grad_norm clipping
                if not config['no_grad_clip']:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()

            if distributed:
                if config['use_vgg']:
                    loss_vgg = reduce_tensor(loss_vgg, config['world_size'])

            if rank == 0:
                if cnt % 10 == 0:
                    run_visualization(output_dict, output_recon, target, input_a, input_b, save_path, tb_logger, cnt)

                print_dict = {"learning_rate": get_learning_rate(optimizer_G)}
                if config['use_vgg']:
                    tb_logger.add_scalar('vgg.loss', loss_vgg, cnt)
                    print_dict['Loss_VGG'] = loss_vgg.data
                if config['use_gan']:
                    tb_logger.add_scalar('gan.loss', loss_G_GAN, cnt)
                    tb_logger.add_scalar('d_real.loss', loss_D_real, cnt)
                    tb_logger.add_scalar('d_fake.loss', loss_D_fake, cnt)
                    print_dict['Loss_G_GAN'] = loss_G_GAN
                    print_dict['Loss_real'] = loss_D_real.data
                    print_dict['Loss_fake'] = loss_D_fake.data

                log_iter(ep, cnt % len(train_dataloader), len(train_dataloader), print_dict, log_handle=log_handle)

            if loss_G != loss_G:
                print("NaN!!")
                exit(-2)

            cnt = cnt+1
            # end of train iter loop
        if ep % config['val_freq'] == 0 and config['val_freq'] > 0:
            val_loss = run_val(model, validationLoss, val_dataloader, os.path.join(save_path, 'val_%d_renders' % (ep)))

            if distributed:
                val_loss = reduce_tensor(val_loss, config['world_size'])
            if rank == 0:
                tb_logger.add_scalar('validation.loss', val_loss, ep)
                log_iter(ep, 1, 1, {"Loss_VGG": val_loss}, header="Validation loss: ", log_handle=log_handle)

        if rank == 0:
            if (ep % config['save_freq'] == 0):
                fname = 'checkpoint_%d.ckpt' % (ep)
                fname = os.path.join(save_path, fname)
                print("Saving model...")
                save_weights(model, fname, distributed)
                optimizer_g_fname = os.path.join(save_path, 'latest_optimizer_g_state.ckpt')
                torch.save(optimizer_G.state_dict(), optimizer_g_fname)
                if config['use_gan']:
                    fname = 'checkpoint_D_%d.ckpt' % (ep)
                    fname = os.path.join(save_path, fname)
                    save_weights(discriminator, fname, distributed)
                    optimizer_d_fname = os.path.join(save_path, 'latest_optimizer_d_state.ckpt')
                    torch.save(optimizer_D.state_dict(), optimizer_d_fname)

        # end of epoch loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, help="Path to config file for current experiment")
    #  defaults.yaml stores list of all options with their default values
    #  do not edit that file unless you're adding additional options or wish to change defaults.
    config = parse_all_args(parser, 'configs/defaults.yaml')
    main(config)
