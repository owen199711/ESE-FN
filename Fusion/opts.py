import argparse
from skeleton_utils import str2bool

def parse_opts():
    parser = argparse.ArgumentParser()
    #train
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--test_batch_size')
    parser.add_argument('--num_epoch')
    parser.add_argument('--n_workers', default=16, type=int, help='Number of workers for dataloader')
    parser.add_argument('--config', default='./config/local/fusion/fusion_two.ymal')
    #config/local/nturgbd-cross-subject/train_joint.yaml
    parser.add_argument('--phase', default='train', help='must be train or test')

    #backbone-RGB
    parser.add_argument('--backbone', default='ResNeXt101', type=str)
    parser.add_argument('--rgb_finetune_path', default='', type=str, help='finetune rgb_model (.pth)')
    parser.add_argument('--resume_path1')
    parser.add_argument('--rgb_model', default='resnext', type=str, help='Model base architecture')
    parser.add_argument('--model_depth', default=50, type=int, help='Number of layers in skeleton_model')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--ft_begin_index', default=4, type=int, help='Begin block index of fine-tuning')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration o8f inputs')
    parser.add_argument('--training', action='store_true', help='training/testing')
    parser.set_defaults(training=True)
    parser.add_argument('--freeze_BN', action='store_true', help='freeze_BN/testing')
    parser.set_defaults(freeze_BN=False)

    #skeleton
    parser.add_argument('--skeleton_mode',default='joint')
    parser.add_argument('--fusion_pretrain',default='',type=str)
    parser.add_argument('--frame_dir', default='E:\\dataSet\\ETRI\\RGB_P001-P010\\frame', type=str,help='path of jpg files')
    parser.add_argument('--train_lable_path',default='',type=str)
    parser.add_argument('--val_lable_path', default='', type=str)
    parser.add_argument('--split', default=1, type=str, help='(for HMDB51 and UCF101)')
    parser.add_argument('--modality', default='RGB', type=str, help='(RGB, skeleton)')
    parser.add_argument('--input_channels', default=3, type=int, help='(3, 2)')
    parser.add_argument('--n_classes', default=55, type=int, help='Number of classes (ETRI:55, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=55, type=int)
    parser.add_argument('--num_segments', default=16, type=int)
    parser.add_argument('--start_epoch',default=0)

    parser.add_argument('--lr_plan', default={11: 0.001, 21: 0.0005, 121: 1e-05})
    parser.add_argument('--output_layers', action='append', help='layer to output on forward pass')
    parser.set_defaults(output_layers=[])

    #dataset
    parser.add_argument('--dataset', default='ETRI', type=str, help='(ETRI, NTU, HMDB51)')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--flow_prefix', default="", type=str)
    parser.add_argument('--train_list', default="E:/dataSet/ETRI/64/train.txt", type=str)
    parser.add_argument('--val_list', default="E:/dataSet/ETRI/64/val.txt", type=str)

    # optimizer parameters
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_patience', default=10, type=int)
    parser.add_argument('--n_epochs', default=140, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int)

    parser.add_argument('--result_path', default='./work_dir', type=str, help='result_path')
    parser.add_argument('--log', default=1, type=int, help='Log training and validation')
    parser.add_argument('--checkpoint', default=2, type=int,help='Trained skeleton_model is saved at every this epochs.')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--random_seed', default=1, type=bool, help='Manually set random seed of sampling validation clip')
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')

    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--work-dir', default='./work_dir/temp')


    parser.add_argument('--show-topk',type=int,default=[1, 5],nargs='+')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--skeleton_model',default='', type=str, help='the skeleton_model will be used')
    parser.add_argument('--model_args',type=dict,default=dict())
    parser.add_argument('--joint_weights',default='')
    parser.add_argument('--bone_weights', default='')
    parser.add_argument('--train-feeder-args',default=dict())
    parser.add_argument('--test-feeder-args',default=dict())

    # optim
    parser.add_argument('--base-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step',type=int, default=[20, 40, 60],nargs='+')
    parser.add_argument('--device',type=int,default=0,nargs='+')
    parser.add_argument('--optimizer',default='SGD', help='type of optimizer')
    parser.add_argument('--warm_up_epoch', default=0)

    parser.add_argument('--weight-decay',type=float,default=0.0005)

    parser.add_argument('--mod', default='RGB',type=str)


    return parser

