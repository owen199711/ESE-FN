from __future__ import print_function
from torch.autograd import Variable
from skeleton_utils import *
from utils import *
from opts import parse_opts

#训练骨骼骨干网络
def train(arg, model,loss_fun,optimizer,epoch,device):
    model.train()
    loader = load_data(arg)['train']
    adjust_learning_rate(arg,epoch,optimizer)
    process = tqdm(loader)
    loss_meter = AverageMeter()
    acc_meter=AverageMeter()

    for batch_idx, (data, label, index) in enumerate(process):
        data = data.to(device)
        label =label.to(device)
        output = model(data)
        loss = loss_fun(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value, predict_label = torch.max(output.data, 1)
        acc = torch.mean((predict_label == label.data).float())
        loss_meter.update(loss,label.size(0))
        acc_meter.update(acc,label.size(0))

    train_info={
        'epoch':epoch,
        'acc':acc_meter.avg,
        'loss':loss_meter.avg
    }
    return train_info

def val(arg,model,loss_fun,epoch,device):
    model.eval()
    val_load=load_data(arg)['test']
    loss_meter=AverageMeter()
    acc_meter1=AverageMeter()
    acc_meter5=AverageMeter()
    process = tqdm(val_load)
    for batch_idx, (data, label, index) in enumerate(process):
        data = data.to(device)
        label =label.to(device)
        with torch.no_grad():
            output = model(data)
            loss = loss_fun(output, label)
            acc_top1=calculate_accuracy(output,label)
            acc_top5=calculate_accuracy5(output,label)
            loss_meter.update(loss.item(), label.size(0))
            acc_meter1.update(acc_top1, label.size(0))
            acc_meter5.update(acc_top5,label.size(0))
    val_info={
        'epoch':epoch,
        'loss':loss_meter.avg,
        'top1':acc_meter1.avg,
        'top5':acc_meter5.avg
    }
    return val_info

def draw(arg,model,loss_fun,device):
    model.eval()
    val_load=load_data(arg)['test']
    loss_meter=AverageMeter()
    acc_meter1=AverageMeter()
    acc_meter5=AverageMeter()
    process = tqdm(val_load)
    pred_buf=[]
    lable_buf=[]
    for batch_idx, (data, label, index) in enumerate(process):
        data = Variable(data.float().to(device),requires_grad=False,volatile=True)
        label = Variable(label.long().to(device),requires_grad=False,volatile=True)

        output = model(data)
       # loss = loss_fun(output, label)
        acc_top1=calculate_accuracy(output,label)
        acc_top5=calculate_accuracy5(output,label)
        #loss_meter.update(loss.item(), label.size(0))
        acc_meter1.update(acc_top1, label.size(0))
        acc_meter5.update(acc_top5,label.size(0))

        _, pred = output.topk(1, 1, True)
        pred = pred.t().view(-1).cpu().numpy()
        labels = label.cpu().numpy()
        for i in range(len(pred)):
            pred_buf.append(pred[i])
            lable_buf.append(labels[i])

    print('top1:{:.2f}% '.format(acc_meter1.avg))
    return pred_buf, lable_buf

def start(arg,device):
    model,loss_fun=load_model(arg,arg.skeleton_mode)
    optimizer=optim.SGD(
            model.parameters(),
            lr=arg.learning_rate,
            momentum=arg.momentum,
            weight_decay=arg.weight_decay,
            nesterov=arg.nesterov)
    best_acc=0
    states = {'epoch': 0, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    log_train_buf=[]
    log_val_buf=[]
    if arg.phase == 'train':
        best_epoch = 0
        for epoch in range(arg.start_epoch, arg.num_epoch):
            train_info=train(arg, model,loss_fun,optimizer,epoch,device)
            str='epoch:{} acc:{} loss:{}'.format(train_info['epoch'],train_info['acc'],train_info['loss'])
            log_train_buf.append(str)
            val_info=val(arg,model,loss_fun,epoch,device)
            temp='epoch:{} top1:{} top5:{} loss:{}'.format(val_info['epoch'],val_info['top1'],val_info['tpo5'],val_info['loss'])
            log_val_buf.append(temp)
            if(val_info['top1']>best_acc):
                best_acc=val_info['top1']
                best_epoch=epoch
                model_saved_name=arg.model_saved_name+str(epoch)+"_"+str(int(best_acc*100)/100)+'.pt'
                path=os.path.join(arg.result_path,arg.dataset,arg.mod)
                states['epoch']=epoch
                states['state_dict']=model.state_dict()
                states['optimizer']=optimizer.state_dict()
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(states,path+'/'+model_saved_name)
            print('eopch:{}/{} best result:{:.3f}% top1:{:.3f}% top5:{:.3f}% loss:{:.3f}'.format(epoch,best_epoch, best_acc * 100,
                                                                                               val_info['top1'] * 100, val_info['top5'] * 100,val_info['loss']))
        opt.log_path = os.path.join(opt.result_path, opt.dataset, opt.mod)
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        log(opt.log_path + '/train_log.txt', log_train_buf)
        log(opt.log_path + '/val_log.txt', log_val_buf)
    elif arg.phase == 'test':
        val_info=eval(arg,model,loss_fun,0,device)
        print('top1:{:.3f}% top5:{:.3f}% loss:{:.3f}'.format(val_info['top1'] * 100, val_info['top5'] * 100,val_info['loss']))
    else:
        dir = opt.rgb_save + '/sk'
        pred, lable = draw(arg,model,loss_fun,device)
        np.savez(dir, pred=pred, lable=lable)

if __name__ == '__main__':
    parser =  parse_opts()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # load arg form config file
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
    opt=arg
    print("lr = {} \t momentum = {} \t batch_size = {} \t sample_duration = {}, \t mod = {}, \t model_size={}"
          .format(opt.learning_rate, opt.momentum, opt.batch_size, opt.sample_duration, opt.modality, opt.model_depth))
    start(arg,device)
