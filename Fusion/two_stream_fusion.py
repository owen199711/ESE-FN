from torch.autograd import Variable
from utils import *
from skeleton_utils import *
from RGB_model.model import generate_model,generate_vgg
from fusion_model.model import generate_fusion
from torch import optim
import numpy as np

#训练融合网络
def train(arg, skeleton_model,RGB_model,fusion,loss_fun,optimizer,epoch,device,loader):
    skeleton_model.eval()
    RGB_model.eval()
    fusion.train()
    adjust_learning_rate(arg,epoch,optimizer)
    process = tqdm(loader)
    losses = AverageMeter()
    accuracies = AverageMeter()
    epoch_timer = Timer()

    for batch_idx, (RGB_data,skeleton_data,label, index) in enumerate(process):
        #get data
        RGB_data =RGB_data.to(device)
        skeleton_data =skeleton_data.to(device)

        label = label.to(device)

        # forward
        RGB_output = RGB_model(RGB_data)
        skeleton_output=skeleton_model(skeleton_data)
        input = torch.cat((RGB_output[1], skeleton_output[1]), dim=1)

        # output = fusion(RGB_output[1],skeleton_output[1])
        output = fusion(input)
        loss_c = loss_fun(output, label)
        loss_r=loss_fun(RGB_output[0],label)
        loss_s=loss_fun(skeleton_output[0],label)

        loss=0.7*loss_c+0.3*(min(loss_r,loss_s)-loss_c)

        acc = calculate_accuracy(output, label)

        losses.update(loss.data, label.size(0))
        accuracies.update(acc, label.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_info={
        'epoch':epoch,
        'acc':accuracies.avg,
        'loss':losses.avg,
        'time':epoch_timer.timeit()
    }
    return train_info

def val(skeleton_model,RGB_model,fusion,loss_fun,epoch,device,val_load):
    fusion.eval()
    skeleton_model.eval()
    RGB_model.eval()
    loss_meter=AverageMeter()
    acc_meter1=AverageMeter()
    acc_meter5=AverageMeter()
    epoch_timer = Timer()
    process = tqdm(val_load)
    for batch_idx, (RGB_data,skeleton_data,label, index) in enumerate(process):
        RGB_data = RGB_data.float().to(device)
        skeleton_data =skeleton_data.float().to(device)
        label = label.long().to(device)
        with torch.no_grad():
            output_r = RGB_model(RGB_data)
            output_s = skeleton_model(skeleton_data)
            input = torch.cat((output_r[1], output_s[1]), dim=1)
            # output = fusion(output_r[1],output_s[1])
            output = fusion(input)
            loss = loss_fun(output, label)
            acc_top1 = calculate_accuracy(output, label)
            acc_top5 = calculate_accuracy5(output, label)
            loss_meter.update(loss.item(), label.size(0))
            acc_meter1.update(acc_top1, label.size(0))
            acc_meter5.update(acc_top5,label.size(0))
    val_info={
        'epoch':epoch,
        'loss':loss_meter.avg,
        'top1':acc_meter1.avg,
        'top5':acc_meter5.avg,
        'time' :epoch_timer.timeit()
    }
    return val_info

def draw(skeleton_model, RGB_model, fusion, loss_fun, epoch, device, val_load):
    fusion.eval()
    skeleton_model.eval()
    RGB_model.eval()
    loss_meter = AverageMeter()
    acc_meter1 = AverageMeter()
    acc_meter5 = AverageMeter()
    epoch_timer = Timer()
    process = tqdm(val_load)
    pred_buf=[]
    lable_buf=[]
    for batch_idx, (RGB_data, skeleton_data, label, index) in enumerate(process):
        RGB_data = Variable(RGB_data.float().to(device), requires_grad=False, volatile=True)
        skeleton_data = Variable(skeleton_data.float().to(device), requires_grad=False)
        label = Variable(label.long().to(device), requires_grad=False, volatile=True)

        with torch.no_grad():
            output_r = RGB_model(RGB_data)
            output_s = skeleton_model(skeleton_data)
            input = torch.cat((output_r[1], output_s[1]), dim=1)
            output = fusion(input)

            _, pred = output.topk(1, 1, True)
            pred = pred.t().view(-1).cpu().numpy()
            labels=label.cpu().numpy()
            for i in range(len(pred)):
                pred_buf.append(pred[i])
                lable_buf.append(labels[i])
            loss = loss_fun(output, label)
            acc_top1 = calculate_accuracy(output, label)
            acc_top5 = calculate_accuracy5(output, label)
            loss_meter.update(loss.item(), label.size(0))
            acc_meter1.update(acc_top1, label.size(0))
            acc_meter5.update(acc_top5, label.size(0))
    val_info = {
        'epoch': epoch,
        'loss': loss_meter.avg,
        'top1': acc_meter1.avg,
        'top5': acc_meter5.avg,
        'time': epoch_timer.timeit()
    }
    print('top1:{:.2f}% top5:{:.2f}% loss:{:.2f}'.format(val_info['top1'] * 100, val_info['top5'] * 100,
                                                         val_info['loss']))
    print('finish')

    return pred_buf,lable_buf

def start(arg,device,loss_fun):
    skeleton_model,_=load_model(arg,arg.skeleton_mode)
    if(arg.backbone=='VGG'):
        RGB_model,_=generate_vgg(arg)
    else:
        RGB_model,_=generate_model(arg)
    arg.is_fusion=True
    fusion=generate_fusion(arg)
    parameters=fusion.parameters()

    train_loader,val_loader=load_data_2(arg)['train'],load_data_2(arg)['test']

    if arg.dataset == 'NTU':
        arg.n_classes = 60
    elif arg.dataset == 'ETRI':
        arg.n_classes = 55

    if arg.resume_path1!="":
        print('loading checkpoint {}'.format(arg.resume_path1))
        checkpoint = torch.load(arg.resume_path1)
        RGB_model.load_state_dict(checkpoint['state_dict'])
    if arg.fusion_pretrain!='':
        arg.weight_decay = 1e-4
        arg.base_lr = 0.025

    optimizer=optim.SGD(
            parameters,
            lr=arg.learning_rate,
            momentum=arg.momentum,
            weight_decay=arg.weight_decay,
            nesterov=arg.nesterov)
    best_acc=0
    states = {'epoch': 0, 'state_dict': fusion.state_dict(), 'optimizer': optimizer.state_dict()}
    log_train_buf = []
    log_val_buf = []
    if arg.phase == 'train':
        best_epoch = 0
        for epoch in range(arg.start_epoch, arg.num_epoch):

            train_info=train(arg, skeleton_model,RGB_model,fusion,loss_fun,optimizer,epoch,device,train_loader)
            st = 'epoch:{} acc:{} loss:{} time:{}'.format(train_info['epoch'], train_info['acc'], train_info['loss'],
                                                           train_info['time'])
            log_train_buf.append(st)

            val_info = val(skeleton_model,RGB_model,fusion,loss_fun, epoch, device,val_loader)
            temp='epoch:{} top1:{} top5:{} loss:{} time:{}'.format(val_info['epoch'],val_info['top1'],val_info['top5'],val_info['loss'],val_info['time'])
            log_val_buf.append(temp)
            if(val_info['top1']>best_acc):
                best_acc=val_info['top1']
                best_epoch=epoch
                model_saved_name=arg.model_saved_name+str(epoch)+"-"+str(int(best_acc*100)/100)+'.pt'
                print('best accuracy: ', best_acc, ' model_name: ', model_saved_name)
                path=os.path.join(arg.result_path,arg.dataset,arg.mod)
                states['epoch']=epoch
                states['state_dict']=fusion.state_dict()
                states['optimizer']=optimizer.state_dict()
                if not os.path.exists(path):
                    os.makedirs(path)
                path=os.path.join(path,model_saved_name)
                torch.save(states,path)

            print('eopch:{}/{} best result:{:.3f}% top1:{:.3f}% top5:{:.3f}% loss:{:.3f}'.format(epoch, best_epoch,
                                                                                                 best_acc * 100,
                                                                                                 val_info['top1'] * 100,
                                                                                                 val_info['top5'] * 100,
                                                                                                 val_info['loss']))

        opt.log_path = os.path.join(opt.result_path, opt.dataset, opt.mod)
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        log(opt.log_path + '/train_log.txt', log_train_buf)
        log(opt.log_path + '/val_log.txt', log_val_buf)
    elif arg.phase == 'test':
        val_info=val(skeleton_model,RGB_model,fusion,loss_fun, 0, device,val_loader)
        print('top1:{:.2f}% top5:{:.2f}% loss:{:.2f}'.format(val_info['top1']*100, val_info['top5']*100, val_info['loss']))
    else:
        preds,lables = draw(skeleton_model, RGB_model, fusion, loss_fun, 0, device, val_loader)
        np.savez(arg.confuse_dir,pred=preds,lable=lables)

if __name__=="__main__":
    parser = parse_opts()
    loss_fun = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        loss_fun=loss_fun.cuda()
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
    init_seed(0)
    opt = arg
    print("lr = {} \t momentum = {} \t batch_size = {} \t sample_duration = {}, \t mod = {}, \t model_size={}"
          .format(opt.learning_rate, opt.momentum, opt.batch_size, opt.sample_duration, opt.modality, opt.model_depth))
    start(arg,device,loss_fun)
    opt = parse_opts()
