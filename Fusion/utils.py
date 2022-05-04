from __future__ import division
import csv
import numpy as np
import pdb
import os
from transforms import *
import cv2
from numpy.random import randint
from opts import parse_opts
import time
import matplotlib.pyplot as plt
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

def readtxt(dir):
    with open(dir,'r') as f:
        for row in f.readlines():
            row=row.strip('\n')
            print(row)

def write(dir,str):
    with open(dir,'a+') as f:
        f.write(str+"\n")
        f.flush()
        f.close()


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size
    
def calculate_accuracy5(outputs, targets):
    batch_size = targets.size(0)
    maxk = max((1, 5))
    y_resize = targets.view(-1, 1)
    _, pred = outputs.topk(maxk, 1, True, True)
    correct = torch.eq(pred, y_resize).sum().float().item()

    return correct / batch_size

def sample_indices(num_frames,length,num_segments):
    """
    :return: list
    """
    average_duration = (num_frames - length + 1) // num_segments  #size for each segment
    if average_duration > 0:
         offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration, size=num_segments)
    elif num_frames > num_segments:
        offsets = np.sort(randint(num_frames - length + 1, size=num_segments))
    else:
        temp=[i for i in range(num_frames)]
        len=num_segments-num_frames
        for i in range(len):
            temp.append(num_frames-1)
        offsets=np.asarray(temp)
    return offsets + 1

def log(file_path,args):
    with open(file_path, 'a+') as f:
        for i in args:
            f.write(i+'\n')
            f.flush()
        f.close()


def get_val_indices(num_frames,length,num_segments):
    if num_frames > num_segments + length - 1:
         tick = (num_frames - length + 1) / float(num_segments)
         offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    else:
        temp = [i for i in range(num_frames)]
        len = num_segments - num_frames
        for i in range(len):
            temp.append(num_frames - 1)
        offsets = np.asarray(temp)
    return offsets + 1

def get_test_indices(num_frames,length,num_segments):

    tick = (num_frames - length + 1) / float(num_segments)

    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])

    return offsets + 1

def getInt(str):
    if(str[-2]=='0'):
        return int(str[-1])
    else:
        return int(str[-2:])

def getlen(s_path,f_path):
    if(not os.path.exists(f_path)):
        print(f_path)
        return 0
    with open(s_path) as f:
        f_len = len([fname for fname in os.listdir(f_path) if fname.endswith('.jpg')])
        content=f.readlines()
        last=content[-1].split(',')
        #print(last,"\n")
        if(last[0]!='' and last[0]!='frameNum'):
            s_len=int(last[0])
        else:
            s_len=len(content)
        if(f_len>s_len):
            if(s_len==1):
                print(s_path)
                s_len=0
            return s_len
        else:
            return f_len


#分割测试训练集
#将数据集划分为训练集和测试集分别存储
#存储格式：类名+总帧数+类别标签+帧列表。
def ETRI_skeleton_train_val(data_dir,save_dir,frame_nums):
    train_data=[]
    val_data=[]
    frame_root='/home/10401006/dataSet/ETRI/frame/'
    p_lists=[i for i in os.listdir(data_dir)]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in p_lists:
        v_name = i.strip('.csv').split('_')
        p_id = getInt(v_name[1])
        a_id = getInt(v_name[0])
        s_path = os.path.join(data_dir, i)
        f_path = os.path.join(frame_root,v_name[1],i.strip('.csv'))
        frame_len = getlen(s_path,f_path)
        if frame_len==0:
            continue
        if (p_id % 3 == 0):
            frame_index = get_val_indices(frame_len,1,frame_nums)
            context = v_name[1]+'/'+i.strip('.csv') + "," + str(frame_len) + "," + str(a_id) + "," + str(frame_index)
            val_data.append(context)
        else:
            frame_index = sample_indices(frame_len, 1, frame_nums)
            context = v_name[1]+'/'+i.strip('.csv') + "," + str(frame_len) + "," + str(a_id) + "," + str(frame_index)
            train_data.append(context)
    train_set_dir = save_dir + '/' + str(frame_nums) + '/' + 'train.txt'
    val_set_dir = save_dir + '/' + str(frame_nums) + '/' + 'val.txt'
    with open(train_set_dir,'a+') as f:
        for i in train_data:
            f.writelines(i+'\n')
            f.flush()
        f.close()
    with open(val_set_dir,'a+') as f:
        for i in val_data:
            f.writelines(i+'\n')
            f.flush()
        f.close()
    print('finish')



class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time
data_dir='D:/BaiduNetdiskDownload/P001-P050'
save_dir='D:/BaiduNetdiskDownload'

def virolize_data(data_dir):
    data_x=[]
    data_y=[]
    with open(data_dir,'r') as f:
        content=f.readlines()[1:]
        i=0
        for row in content:
            if i%1==0:
                row = row.strip('\n').split('\t')
                data_x.append(row[0])
                data_y.append(row[-1])
            i+=1;

        plt.plot(data_x,data_y)
        plt.show()

def tolist(frames):
    array = frames.strip('[').strip("\n").strip(']').split(' ')
    restult = []
    for i in array:
        if (i != '[' and i != '' and i != ']'):
            restult.append(int(i))
    return restult

def store_image(data_dir,index,save_dir):
    for i in index:
        im_p=os.path.join(data_dir, '{:05d}.jpg'.format(i))
        try:
          img=cv2.imread(im_p)
        except:
          print(im_p)
        # temp=save_dir+"/"+'{:05d}.jpg'.format(i)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # cv2.imwrite(temp,img)

def move_data(data_dir,save_dir,frame_path):
    with open(data_dir, 'r') as f:
        content = ''
        for row in f.readlines():
            row = row.strip('\n')
            if (row[-1] != ']'):
                content += row
            else:
                content += row
                conte=content.strip('\n').split(',')
                frame_index = tolist(conte[3])
                im_path=frame_path+"/"+conte[0]
                temp=save_dir+"/"+conte[0]
                store_image(im_path,frame_index,temp)
                content=''

def extract_1(vid_dir, frame_dir, start, end):
  p_path_dir=os.listdir(vid_dir)[start:end]
  for row in p_path_dir:
      vid_path=vid_dir+'/'+row
      outdir = os.path.join(frame_dir,row[:-4])
      try:
          os.system('mkdir -p "%s"' % (outdir))
          o = subprocess.check_output(
              'ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"' % (
                  vid_path), shell=True).decode("utf-8")
          lines = o.splitlines()
          width = int(lines[0].split('=')[1])
          height = int(lines[1].split('=')[1])
          resize_str = '-1:256' if width > height else '256:-1'
          os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s" > /dev/null 2>&1' % (
              vid_path, resize_str, os.path.join(outdir, '%05d.jpg')))
          nframes = len([fname for fname in os.listdir(outdir) if fname.endswith('.jpg')])
          if nframes == 0: raise Exception
          os.system('touch "%s"' % (os.path.join(outdir, 'done')))
      except:
          print("ERROR", vid_path)

def getlen_ntu(s_path,f_path):
    if(not os.path.exists(f_path)):
        print(f_path)
        return 0
    with open(s_path) as f:
        f_len = len([fname for fname in os.listdir(f_path) if fname.endswith('.jpg')])
        s_len=int(f.readlines()[0])
        return f_len if f_len<s_len else s_len

def getRGB_train_val_on_NTU(label_path,frame_path,skeleton_path,save_path,sample_num):
    with open(label_path, 'rb') as f:
        sample_name, label = pickle.load(f, encoding='latin1')
        res=[]
        for i in sample_name:
            name=i.strip('.skeleton')+'_rgb'
            f_path=frame_path+'/'+name
            s_path=skeleton_path+'/'+i
            f_len=getlen_ntu(s_path,f_path)
            frames_list=sample_indices(f_len,1,sample_num)
            content=name+','+str(frames_list)
            print(content)
            res.append(content)
        with open(save_path,'a+') as f:
            for i in res:
                f.writelines(i+'\n')
                f.flush()
            f.close()


def clip(data_dir,save_dir):
    with open(data_dir,'r') as f:
        contents=f.readlines()
        young_buf=[]
        elderly_buf=[]
        content=''
        for i in contents:
            row = i.strip('\n')
            if (row[-1] != ']'):
                content += row
            else:
                content += row
                pid = getInt(content.split('/')[0])
                if (pid >= 50):
                    young_buf.append(i)
                else:
                    elderly_buf.append(i)
                content=''

    for i in young_buf:
        write(save_dir+'_young.txt',i.strip('\n'))
    for i in elderly_buf:
        write(save_dir+'_elderly.txt',i.strip('\n'))





if __name__=='__main__':
   # data_dir='D:/RGB_dataset/1.log'
   # virolize_data(data_dir)
   d1='/home/10401006/dataSet/ETRI/64/train.txt'
   d2='/home/10401006/dataSet/ETRI/64/train'
   clip(d1,d2)
   # frame_dir='/home/10401006/dataSet/ETRI/frame/'
   # save_dir='/home/user01/dataSet/etri/frame_16/'
   # data_dir='/home/user01/dataSet/etri/16/train.txt'
   # d1='D:\\code\\Cross_MARS\\ETRI\\skeleton\\120_0.88.pt'
   # vid_dir="E:\\video"
   frame_dir="/home/10401006/dataSet/NTU/frame"
   lable_path='/home/10401006/dataSet/NTU/xview/val_label.pkl'
   skeleton_path='/home/10401006/dataSet/NTU/skeleton/nturgb+d_skeletons'
   save_path='/home/10401006/dataSet/NTU/frame_train_val_index/cv/val.txt'
   getRGB_train_val_on_NTU(lable_path,frame_dir,skeleton_path,save_path,64)
   #extract_1(vid_dir, frame_dir, 1, 500)
   #change(d1,d1)
   #move_data(data_dir,save_dir,save_dir)
   print('finish')
   #ETRI_skeleton_train_val(data_dir,save_dir,128)