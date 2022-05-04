from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from RGB_model.model import generate_model,generate_vgg
from torch.autograd import Variable
from dataset.rgb_dataset import *
from dataset.ntu_dataset import *
from utils import *
import yaml
from skeleton_utils import *

#训练RGB骨干网络
def train_net(opt,device,criterion):
    torch.manual_seed(opt.manual_seed)
    if opt.mod == 'RGB':
        opt.input_channels = 3
        data_length = 1
    if (opt.dataset == 'ETRI'):
        train_data = TSNDataSet(opt.train_list, data_length, opt, 'train', modality=opt.mod, image_tmpl="{:05d}.jpg")
        val_data = TSNDataSet(opt.val_list, data_length, opt, 'test', modality=opt.mod, image_tmpl="{:05d}.jpg")
    else:
        train_data = DataSet(opt.train_list,opt.train_lable_path,data_length, opt, 'train', modality=opt.mod, image_tmpl="{:05d}.jpg")
        val_data = DataSet(opt.val_list,opt.val_lable_path, data_length, opt, 'test', modality=opt.mod, image_tmpl="{:05d}.jpg")


    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers,
                                pin_memory=True, drop_last=True)
    print("Length train datat = ", len(train_dataloader))
    print("Length validation data = ", len(val_dataloader))

    # define the skeleton_model
    print("Loading RGB_model... ", opt.backbone, opt.model_depth)
    if(opt.backbone=='VGG'):
        model, parameters = generate_vgg(opt)
    else:
        model, parameters = generate_model(opt)

    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)  #
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # if opt.pretrain_path:
    #     opt.weight_decay = 1e-5
    #     opt.learning_rate = 0.001

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    if opt.resume_path1 != '':
        optimizer.load_state_dict(torch.load(opt.resume_path1)['optimizer'])

    best_result = {'epoch': 0, 'acc': 0}
    start_epoch = 1

    states = {'epoch': 0, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    val_log_buf=[]
    train_log_buf=[]
    if opt.phase=='train':
        for epoch in range(start_epoch, start_epoch + opt.n_epochs):
            train_info = train(train_dataloader, model, device, optimizer, epoch, criterion)
            str='epoch:{} acc:{} loss:{} time:{}'.format(train_info['epoch'],train_info['acc'],train_info['loss'],train_info['time'])
            train_log_buf.append(str)
            # Test
            if epoch % opt.test_interval_epoch == 0:
                test_info = val(val_dataloader, model, device, epoch, criterion)
                temp='epoch:{} acc:{} loss:{} time:{}'.format(test_info['epoch'],test_info['acc'],test_info['loss'],test_info['time'])
                val_log_buf.append(temp)
                if test_info['acc'] > best_result['acc']:
                    if opt.pretrain_path:
                        save_model_path = os.path.join(opt.log_path,
                                                       'preKin_{}_{}_epoch:{}_acc:{}.pth'
                                                       .format(opt.mod, opt.dataset, epoch, test_info['acc']))
                    else:
                        save_model_path = os.path.join(opt.log_path,
                                                       '{}_{}_epoch:{}_acc:{}.pth'.format(opt.mod, opt.dataset, epoch, test_info['acc']))
                    best_result = test_info
                    states['epoch'] = epoch + 1
                    states['state_dict'] = model.state_dict()
                    states['optimizer'] = optimizer.state_dict()

                    torch.save(states, save_model_path)
                print('epoch:{} acc:{:.2f}%  loss:{:.2f}'.format(epoch, test_info['acc'], test_info['loss']))
                print('best epoch:{} best acc:{:.2f}% loss:{:.2f}'.format(best_result['epoch'], best_result['acc'],
                                                                          test_info['loss']))

        #log
        opt.log_path = os.path.join(opt.result_path, opt.dataset, opt.mod)
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        log(opt.log_path+'/train_log.txt',train_log_buf)
        log(opt.log_path + '/val_log.txt', val_log_buf)
    else :
        dir=opt.rgb_save+'/rgb'
        pred,lable = darw(val_dataloader,model, device, criterion)
        np.savez(dir, pred=pred, lable=lable)

def train(data_loader, model, device, optimizer, epoch, criterion):
    model.train()
    epoch_timer = Timer()
    adjust_learning_rate(arg, epoch, optimizer)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if(i%20==0):print('ith:{} acc:{} loss:{} '.format(i,acc_meter.avg,loss_meter.avg))
    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'acc': acc_meter.avg * 100
    }

    return train_info

def val(val_dataloader, model, device, epoch, criterion):
    model.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_dataloader):
            targets = targets.to(device)
            inputs = inputs.to(device)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs,_ = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))

        test_info = {
            'time': epoch_timer.timeit(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'acc': acc_meter.avg * 100
        }
        return test_info

def darw(val_dataloader,model, device, criterion):
    model.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    pred_buf=[]
    lable_buf=[]
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_dataloader):
            targets = targets.to(device)
            inputs = inputs.to(device)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))
            _, pred = outputs.topk(1, 1, True)
            pred = pred.t().view(-1).cpu().numpy()
            labels = targets.cpu().numpy()
            for i in range(len(pred)):
                pred_buf.append(pred[i])
                lable_buf.append(labels[i])

        print('top1:{:.2f}% loss:{:.2f}'.format(acc_meter.avg,loss_meter.avg))
        return pred_buf,lable_buf

if __name__ == "__main__":
    parser = parse_opts()
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        criterion = criterion.cuda()
    else:
        device = torch.device('cpu')
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    opt = arg
    print("lr = {} \t momentum = {} \t batch_size = {} \t sample_duration = {}, \t mod = {}, \t model_size={}"
          .format(opt.learning_rate, opt.momentum, opt.batch_size, opt.sample_duration, opt.modality, opt.model_depth))
    train_net(opt,device,criterion)










