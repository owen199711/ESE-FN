import pickle
from torch.utils.data import Dataset
from dataset.preprocess_data import *
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.extend(['../'])

from dataset.skeleton_dataset import tools
data_root='E:/dataSet/ETRI/RGB_P001-P010/frame/'
class Feeder(Dataset):
    def __init__(self, joint_data_path, joint_label_path,bone_data_path,bone_label_path,length,
                 opt,train_val,random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, use_mmap=True):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """
        self.joint_data_path = joint_data_path
        self.joint_label_path = joint_label_path
        self.bone_data_path = bone_data_path
        self.bone_label_path = bone_label_path
        self.length = length
        self.opt = opt
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data(joint_data_path,joint_label_path,'joint')
        self.load_data(bone_data_path, bone_label_path,'bone')
        if normalization:
            self.get_mean_map()

    def load_data(self,data_path,lable_path,mod):
        # data: N C V T M
        try:
            with open(lable_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(lable_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        if mod=='joint':
            if self.use_mmap:
                self.joint_data = np.load(data_path, mmap_mode='r',allow_pickle=True)
            else:
                self.joint_data = np.load(data_path,allow_pickle=True)
        else :
            if self.use_mmap:
                self.bone_data = np.load(data_path, mmap_mode='r',allow_pickle=True)
            else:
                self.bone_data = np.load(data_path,allow_pickle=True)

    def get_mean_map(self):
        data = self.joint_data
        N, C, T, V, M = data.shape
        self.joint_mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.joint_std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        data = self.bone_data
        N, C, T, V, M = data.shape
        self.bone_mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.bone_std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def deal_with_data(self,data_numpy,mean_map,std_map):
        if self.normalization:
            data_numpy = (data_numpy - mean_map) / std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        return data_numpy

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        joint_numpy = self.joint_data[index]
        label = self.label[index]
        joint_numpy = np.array(joint_numpy)
        if self.normalization:
          joint_numpy=self.deal_with_data(joint_numpy,self.joint_mean_mapm,self.joint_std_map)

        bone_numpy = self.bone_data[index]
        bone_numpy = np.array(bone_numpy)
        if self.normalization:
           bone_numpy=self.deal_with_data(bone_numpy, self.bone_mean_mapm, self.bone_std_map)
        return joint_numpy,bone_numpy,label


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

