import pickle
from torch.utils.data import Dataset
from dataset.preprocess_data import *
import sys,os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.extend(['../'])

from dataset import tools

class Feeder(Dataset):
    def __init__(self, skeleton_data_path, skeleton_label_path,list_file,length,
                 image_tmpl,opt,train_val,modality='RGB',
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, use_mmap=True):
        """
        :param skeleton_data_path:调用data_gen/gen_etridata.py产生的.npy文件
        :param skeleton_label_path:调用data_gen/gen_etridata.py产生的.pkl文件
        :param list_file:划分的训练测试集文件
        :param length:
        :param image_tmpl:

        """
        self.data_path = skeleton_data_path
        self.label_path = skeleton_label_path
        self.list_file = list_file
        self.length = length
        self.image_tmpl=image_tmpl
        self.opt = opt
        self.data_root=opt.data_root
        self.train_val = train_val
        self.modality = modality
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        self._parse_list()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.skeleton_data = np.load(self.data_path, mmap_mode='r')
        else:
            self.skeleton_data = np.load(self.data_path)


    def get_mean_map(self):
        data = self.skeleton_data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))


    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                path=os.path.join(directory, self.image_tmpl.format(idx))
                im = Image.open(path)
                im2 = im.convert('RGB')
                return [im2]
            except :
                print("Cannot load : {}".format(path))

    def _parse_list(self):
        with open(self.list_file,'r') as f:
            video_list=[]
            content = ''
            for row in f.readlines():
                row=row.strip('\n')
                if (row[-1] != ']'):
                    content+=row
                else:
                    content+=row
                    video_list.append([content.split(',')])
                    content=''
            self.video_list=video_list

    def tolist(self,frames):
        array = frames.strip('[').strip("\n").strip(']').split(' ')
        restult = []
        for i in array:
            if (i != '[' and i != '' and i != ']'):
                restult.append(int(i))
        return restult

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.length):
                seg_imgs = self._load_image(self.data_root+'/'+record[0], p)
                images.extend(seg_imgs)
        att = scale_crop(images, self.train_val, self.opt)
        if (self.opt.backbone == 'VGG'):
            c, b, h, w = att.shape
            att = att.reshape(-1, h, w)
        return att

    def __len__(self):
        return len(self.video_list)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.skeleton_data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        record = self.video_list[index][0]

        frame_index = self.tolist(record[-1])


        RGB_data_numpy=self.get(record, frame_index)


        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return RGB_data_numpy,data_numpy,label, index


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


