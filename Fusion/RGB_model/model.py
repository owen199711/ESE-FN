from __future__ import division
import torch
from torch import nn
from RGB_model import resnext
from RGB_model.vgg16 import VGG16
import pdb
from RGB_model.resnext import get_fine_tuning_parameters


def generate_vgg(opt):
    model=VGG16(64,n_classes=55)
    if opt.pretrain_path:
        print('loading pretrained RGB_model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model = model.cuda()
    model = nn.DataParallel(model)
    return model, model.parameters()

def generate_model(opt):
    assert opt.rgb_model in ['resnext']
    assert opt.model_depth in [101,50,18]

    from RGB_model.resnext import get_fine_tuning_parameters
    if opt.model_depth==18:
        model = resnext.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers)
    elif opt.model_depth==50:
        model = resnext.resnet50(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers)
    else:
        model = resnext.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers)

    model = model.cuda()
    model = nn.DataParallel(model)

    #迁移学习
    if opt.rgb_finetune_path:
        print('loading pretrained skeleton_model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)

        model_dict = model.state_dict()

        state_dict = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict.keys()}

        model_dict.update(state_dict)

        model.load_state_dict(model_dict)

        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()

def generate_fusion(opt):
    #model=fusion_mod1(MyMTMM_3,MyMTMM_3,256,opt.n_finetune_classes)
    #model = fusion_mod4(2,256,opt.n_finetune_classes,16)# -->85.802
    #model = fusion_mod3(my_SE5,SEAttention1, 256, opt.n_finetune_classes)#-->86.08
    #model = fusion_mod3(SEAttention, my_SE5, 256, opt.n_finetune_classes)
    #model = fusion_mod3(my_SE5, MyMTMM_1, 256, opt.n_finetune_classes)
    #model = fusion_mod3(my_SE5, MyMTMM_3, 256, opt.n_finetune_classes)
    #model = fusion_mod3(my_SE5, my_SE5, 256, opt.n_finetune_classes) #(60933)
    #model = fusion_mod3(SEAttention, MyMTMM_3, 256, opt.n_finetune_classes)
    #model = fusion_mod3(my_SE5, my_SE7, 256, opt.n_finetune_classes)
    #model = fusion_mod3(myEA_con, my_SE5, 256, opt.n_finetune_classes)
    #model = fusion_mod3(My_3, My_3, 256, opt.n_finetune_classes)
    #model =fusion_mod1(my_6,256,55)
    #model = fusion_mod3(CPSPPSELayer,SEAttention1, 256, 55)#->86.975
    #model = fusion_mod3(my_9, SEAttention1, 256, 55)87.007
    #model = fusion_mod3(my_9, SEAttention2, 256, 55)
    #model = fusion_mod4(my_9, SEAttention2, 256, 55)
    #model = fusion_mod5(my_9, SEAttention2, 256, 55)
    #model = fusion_mod6(my_9, SEAttention2, 256, 55)
    # model =Fusiontwostream_net(2048,256,55) #-->(65530)
    # model = fusion_mod1(my_9, 256, 55)  # (66801) #M-Net
    # model = fusion_mod2(55) #(66886)->91.96 #sample splicing

    model = fusion_mod3(my_9, SEAttention1, 256, 55) #MC-FFN (66943)->
    # model = fusion_mod4(SEAttention2, 256, 55) #C-Net (66948)
    #model = fusion_mod5(my_9,SEAttention2 , 256, 55) #CM-FFN

   # model = fusion_mod2(my_9, MyMTMM_1, 256, 55)
    #model = fusion_mod3(CPSPPSELayer, SCSEModule, 256, 55)
    #model=nn.DataParallel(model)#gpu 并行
    model=model.cuda()
    if opt.fusion_pretrain!='':
        pretrain = torch.load(opt.fusion_pretrain)

        model_dict = model.state_dict()

        state_dict = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict.keys()}

        model_dict.update(state_dict)

        model.load_state_dict(model_dict)

    return model


