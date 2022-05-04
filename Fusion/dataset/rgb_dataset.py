import torch.utils.data as data

import os
import os.path
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

class TSNDataSet(data.Dataset):
    def __init__(self,list_file,length,opt,train_val,modality='RGB',image_tmpl='img_{:05d}.jpg',random_shift=True):
        self.list_file = list_file #训练测试集
        self.length = length
        self.opt=opt
        self.data_root=opt.data_root
        self.backbone=opt.backbone
        self.train_val=train_val
        self.modality = modality
        self.image_tmpl=image_tmpl
        self.random_shift = random_shift
        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

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
        frame_index=self.tolist(record[3])
        return self.get(record, frame_index)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.length):
                seg_imgs = self._load_image(self.data_root+'/'+record[0], p)
                images.extend(seg_imgs)
                if p < int(record[1]):
                    p += 1
        att=scale_crop(images, self.train_val, self.opt)
        if (self.opt.backbone == 'VGG'):
            c, b, h, w = att.shape
            att = att.reshape(-1, h, w)
        return (att, int(record[2])-1)

    def __len__(self):
        return len(self.video_list)
