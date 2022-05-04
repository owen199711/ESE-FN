from fusion_model.fusionNet import *

def generate_fusion(opt):
    model=Fusion_Net(2048,256,opt.n_finetune_classes)
    if opt.fusion_pretrain!='':
        print('loading pretrained fusion_model {}'.format(opt.fusion_pretrain))
        pretrain = torch.load(opt.fusion_pretrain)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model = model.cuda()
    model = nn.DataParallel(model)
    return model