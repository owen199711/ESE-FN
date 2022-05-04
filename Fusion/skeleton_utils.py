import os
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import time
def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return skeleton_model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def adjust_learning_rate(arg, epoch,optimizer):
    if arg.optimizer == 'SGD' or arg.optimizer == 'Adam':
        if epoch < arg.warm_up_epoch:
            lr = arg.base_lr * (epoch + 1) / arg.warm_up_epoch
        else:
            lr = arg.base_lr * (0.1 ** np.sum(epoch >= np.array(arg.step)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('lr:{}'.format(lr))
        return lr
    else:
        raise ValueError()



def record_time():
    cur_time = time.time()
    return cur_time

def split_time(cur_time):
    split_time = time.time() - cur_time
    record_time()
    return split_time

def load_model(arg,mod):
    output_device = arg.device[0] if type(arg.device) is list else arg.device
    Model = import_class(arg.skeleton_model)
    shutil.copy2(inspect.getfile(Model), arg.work_dir)
    model = Model(**arg.model_args).cuda(output_device)
    loss = nn.CrossEntropyLoss().cuda(output_device)
    if mod=='joint':
        if arg.joint_weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])

            if '.pkl' in arg.joint_weights:
                with open(arg.joint_weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(arg.joint_weights)

            model.load_state_dict(weights['state_dict'])
    else:
        if arg.bone_weights:
            if '.pkl' in arg.bone_weights:
                with open(arg.bone_weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(arg.bone_weights)
            model.load_state_dict(weights['state_dict'])

    if type(arg.device) is list:
        if len(arg.device) > 1:
            model = nn.DataParallel(
                model,
                device_ids=arg.device,
                output_device=output_device)
    return model,loss

def load_model_128(arg,mod):
    output_device = arg.device[0] if type(arg.device) is list else arg.device
    Model = import_class(arg.skeleton_model)
    shutil.copy2(inspect.getfile(Model), arg.work_dir)
    model = Model(**arg.model_args).cuda(output_device)
    loss = nn.CrossEntropyLoss().cuda(output_device)
    if mod=='joint':
        if arg.joint_weights:

            weights = torch.load(arg.joint_weights)
            state_dict = weights['state_dict']
            res = {}
            for _, i in enumerate(state_dict):
                k = i[7:]
                res[k] = state_dict[i]
            state_dict = res

            model.load_state_dict(state_dict)
    else:
        if arg.bone_weights:
            if '.pkl' in arg.bone_weights:
                with open(arg.bone_weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(arg.bone_weights)
            model.load_state_dict(weights['state_dict'])

    if type(arg.device) is list:
        if len(arg.device) > 1:
            model = nn.DataParallel(
                model,
                device_ids=arg.device,
                output_device=output_device)
    return model,loss

def load_optimizer(arg,model):
    if arg.optimizer == 'SGD':
        parameters = model.parameters()
        optimizer = optim.SGD(
            parameters,
            lr=arg.learning_rate,
            momentum=arg.momentum,
            weight_decay=arg.weight_decay,
            nesterov=arg.nesterov)
    elif arg.optimizer == 'Adam':
            optimizer = optim.Adam(
            model.parameters(),
            lr=arg.base_lr,
            weight_decay=arg.weight_decay)
    else:
        raise ValueError()

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                          patience=10, verbose=True,
                                          threshold=1e-4, threshold_mode='rel',
                                          cooldown=0)

    return optimizer,lr_scheduler

def my_collate(batch):
    data = torch.stack([item[0].unsqueeze(0) for item in batch], 0)
    target = torch.Tensor([item[1] for item in batch])
    return [data, target]

def load_data(arg):
    Feeder = import_class(arg.feeder)
    data_loader = dict()
    if arg.phase == 'train':
        data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(**arg.train_feeder_args),
            batch_size=arg.batch_size,
            shuffle=True,
            num_workers=arg.n_workers,
            drop_last=True,
            worker_init_fn=init_seed)
    data_loader['test'] = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.test_feeder_args),
        batch_size=arg.test_batch_size,
        shuffle=False,
        num_workers=arg.n_workers,
        drop_last=False,
        worker_init_fn=init_seed)
    return data_loader

def load_data_2(arg):
    Feeder = import_class(arg.feeder)
    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.train_feeder_args, opt=arg, train_val='train'),
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=arg.n_workers,
        drop_last=True,
        worker_init_fn=init_seed)
    data_loader['test'] = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.test_feeder_args, opt=arg, train_val='test'),
        batch_size=arg.test_batch_size,
        shuffle=False,
        num_workers=arg.n_workers,
        drop_last=False,
        worker_init_fn=init_seed)
    return data_loader