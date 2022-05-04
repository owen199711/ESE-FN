import torch.utils.data as data

import os
import os.path
import pickle
from dataset.preprocess_data import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def frame_index(self):
        return self._data[3]

class DataSet(data.Dataset):
    def __init__(self,list_file,lable_path,length,opt,train_val,modality='RGB',image_tmpl='img_{:05d}.jpg',random_shift=True):
        self.list_file = list_file #训练测试集文件
        self.lable_path=lable_path #.pkl文件
        self.length = length
        self.opt=opt
        self.train_val=train_val
        self.modality = modality
        self.image_tmpl=image_tmpl
        self.random_shift = random_shift
        self._parse_list()
        self.load_data()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.lable_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.lable_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

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
                    video_list.append([content.strip('\n').split(',')])
                    content=''
            self.video_list=video_list

    def tolist(self,frames):
        array = frames.strip('[').strip("\n").strip(']').split(' ')
        restult = []
        for i in array:
            if (i != '[' and i != '' and i != ']'):
                restult.append(int(i))
        return restult

    def __getitem__(self, index):
        record = self.video_list[index][0]
        frame_index=self.tolist(record[1])
        frame_data=self.get(record[0], frame_index)
        lable=self.label[index]
        return frame_data,lable

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            for i in range(self.length):
                seg_imgs = self._load_image(data_root+'/'+record, int(seg_ind))
                images.extend(seg_imgs)
        return scale_crop(images, self.train_val, self.opt)

    def __len__(self):
        return len(self.video_list)
