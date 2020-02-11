import torch
import torch.distributed
import yaml
import os
from models.part_factorized_model import PartFactorizedModel


def denormalize_batch(batch, div_factor=1):
    """denormalize for visualization"""
    # normalize using imagenet mean and std

    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[0, :, :] = 0.5
    mean[1, :, :] = 0.5
    mean[2, :, :] = 0.5
    std[0, :, :] = 0.5
    std[1, :, :] = 0.5
    std[2, :, :] = 0.5
    batch = (batch * std + mean) * div_factor
    return batch


def get_model(args):
    if args['model'] == 'PartFactorizedModel':
        return PartFactorizedModel(args)
    else:
        print("No such model")
        exit(1)


def save_weights(model, save_path, used_DDP=False):
    """Model saving wrapper
    If use_DDP, then save on model.module, else save model directly
    Args:
        model (torch.nn.module): Model to save.
        save_path (str): Path to the checkpoint
        used_DDP (bool): Whether the model was trained with DDP
    """
    if used_DDP:
        #  unwrap the model
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)


def load_weights(ckpt):
    """
    For safety, force-load the model onto the CPU
    """
    state_dict = torch.load(ckpt, map_location=lambda storage, loc: storage)
    return state_dict


def save_options(opt_name, config):
    """saves current model options

    Args:
        opt_name (str): path to options save file
        config (dict): current config dictionary
    """
    with open(opt_name, 'wt') as opt_file:
        # opt_file.write('------------ Options -------------\n')
        for k, v in sorted(config.items()):
            if k in {'use_DDP', 'rank', 'world_size'}:
                continue
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        # opt_file.write('-------------- End ----------------\n')


def log_iter(epoch, iter, total, print_dict, header=None, log_handle=None):
    """
    function for printing out losses, and optionally logs to a logfile if handle is passed in
    """
    msg = "Epoch %d iter %d / %d" % (epoch, iter, total)
    if header is not None:
        msg = header + msg
    for k, v in print_dict.items():
        msg = msg + " | %s: %f" % (k, v)
    if iter % 10 == 0:
        log_handle.flush()
    print(msg)

    if log_handle is not None:
        log_handle.write("%s\n" % msg)


def reduce_tensor(tensor, world_size):
    # reduce tensore for DDP
    # source: https://raw.githubusercontent.com/NVIDIA/apex/master/examples/imagenet/main_amp.py
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def parse_all_args(parser, defaults_file, return_args=False):
    """Processes the command line args from parser

    Processes the command line args stored in the parser to override
    defaults, then stores everything in a dictionary
    Args:
        parser (argparse.ArgumentParser): Argument parser for command line
        defaults_file (str): Path to yaml file storing default values for all options
    Returns:
        config (dict): All options stored in a dictionary
    """
    default_configs_h = open(defaults_file, 'r')
    config = yaml.load(default_configs_h, Loader=yaml.FullLoader)
    default_configs_h.close()

    # add defaults.yaml options to the parser
    for option, value in config.items():
        if type(value) == bool:
            parser.add_argument('--'+option, action='store_true')
        else:
            parser.add_argument('--'+option, type=type(value))

    args = parser.parse_args()
    # read in the specified config
    user_config_h = open(args.config, 'r')
    user_config = yaml.load(user_config_h, Loader=yaml.FullLoader)
    user_config_h.close()
    for option, value in user_config.items():
        config[option] = value

    # now override again if specified in commandline args
    for option in config.keys():
        func = getattr(args, option)
        if func is not None and func is not False:
            config[option] = func

    # set the DDP params
    config['rank'] = 0  # default value
    if config['use_DDP']:
        config['world_size'] = int(os.environ['WORLD_SIZE'])
        config['rank'] = int(os.environ['RANK'])
        config['local_rank'] = int(os.environ['LOCAL_RANK'])
    if return_args:
        return config, args
    else:
        return config


def initialize_distributed(config):
    """
    Sets up necessary stuff for distributed
    training if the world_size is > 1
    Args:
        config (dict): configurations for this run
    """
    if not config['use_DDP']:
        return
    # Manually set the device ids.
    local_rank = config['local_rank']
    world_size = config['world_size']
    rank = config['rank']
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print('Global Rank:')
    print(rank)

    # Call the init process
    if world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
        master_port = os.getenv('MASTER_PORT', '6666')
        init_method += master_ip+':'+master_port
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size, rank=rank,
            init_method=init_method)


def get_learning_rate(optimizer):
    """
    Extracts the optimizer's base learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
