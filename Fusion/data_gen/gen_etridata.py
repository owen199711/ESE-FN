import argparse
import pickle
from tqdm import tqdm
import sys
import csv

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization
training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300
import numpy as np
import os
def get_frames(skeletons,i):
    result=[]
    for row in skeletons[1:]:
        content=row.split(',')
        if(content[0]!=''):
            if int(content[0]) == i:
                result.append(row)
            if int(content[0]) > i:
                return result
    return result

def fun_isdigit(aString):
    for s in aString:
        try:
            float(s)
        except ValueError as e:
            return False
    return True

def read_skeleton_sequence(skeleton_path,frame_index):
    skeleton_sequence = {}
    index1=[]
    value1=[]
    joint_info_key = [
        'x', 'y', 'z', 'depthX', 'depthY',
        'orientationW', 'orientationX', 'orientationY',
        'orientationZ', 'trackingState'
    ]
    with open(skeleton_path,'r') as f:
        skeletons=f.readlines()
        skeleton_sequence['numFrame'] =len(frame_index)
        skeleton_sequence['frameInfo']=[]
        for i in frame_index:
            frame_list=get_frames(skeletons,i)
            index1.append(i)
            value1.append(len(frame_list))
            frame_info = {}
            frame_info['numBody']=len(frame_list)
            frame_info['bodyInfo']=[]
            for row in frame_list:
                skeleton=row.split(',')
                body_info = {}
                body_info['jointInfo'] = []
                if(len(skeleton)>1):
                    body_info['bodyID'] = skeleton[1]
                    body_info['numJoint'] = 25
                    s_que=skeleton[3:]
                    for index in range(body_info['numJoint']):
                        s=s_que[index*10:(index+1)*10]
                        joint_info = {k: 0.0 for k in joint_info_key}
                        if(fun_isdigit(s)==True):
                            joint_info = {
                                k: float(v)
                                for k, v in zip(joint_info_key, s)
                            }
                        body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence,index1,value1

def get_csv_len(data_dir):
    print(data_dir)
    with open(data_dir, 'r') as f:
        reader = f.readlines()
        return int(reader[-1].split(',')[0])

def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  
    else:
        s = 0
    return s

def read_xyz(file,frame_index, max_body=4, num_joint=25):  
    seq_info,index1,value1= read_skeleton_sequence(file,frame_index)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass
    # select two max energy body

    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    data = data.transpose(3, 1, 2, 0)
    return data,index1,value1

def tolist(str):
    array=str.strip('[').strip("\n").strip(']').split(' ')
    restult=[]
    for i in array:
        if(i!='[' and i!='' and i!=']'):
            restult.append(int(i))
    return restult

def gendata(data_path, train_val_dir,out_path,part):
    sample_v_name = []
    sample_v_label = []
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    with open(train_val_dir) as f:
        context=f.readlines()
        data_len=len(os.listdir(data_path))
        fp = np.zeros((data_len, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
        content=''
        index=0
        for i,row in enumerate(context):
            row=row.strip('\n')
            if(row[-1]!=']'):
                content+=row
            else:
                content+=row
                v_content = content.split(',')
                v_name = v_content[0].split('/')[-1]
                frame_index = tolist(v_content[-1])
                sample_v_name.append(v_name)
                sample_v_label.append(int(v_content[-2])-1)
                skeleton_path = data_path + "/" + v_name + ".csv"
                if(os.path.exists(skeleton_path)):
                    print(v_name,"-->",v_content)
                    data,index1,value1 = read_xyz(skeleton_path, frame_index, max_body=max_body_kinect, num_joint=num_joint)
                    print(v_name," ",index1,value1," ",data[0:5])
                    fp[index, :, 0:data.shape[1], :, :] = data
                else :
                    print(skeleton_path,"----not exists!!!!!")
                content=''
                index += 1
        fp = pre_normalization(fp)
        np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_v_name, list(sample_v_label)), f)


if __name__ == '__main__':
    # change(data_dir,save_dir)
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/home/10401006/dataset/ETRI/skeleton/P001-P100/P001-P050')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/home/10401006/dataset/ETRI/')
    parser.add_argument('--train_val_dir', default="/home/10401006/dataset/ETRI/64/")

    benchmark = ['sub']
    part = ['val', 'train']
    arg = parser.parse_args()
    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            train_val_dir=arg.train_val_dir+'{}.txt'.format(p)
            gendata(
                arg.data_path,
                train_val_dir,
                out_path,
                p,
            )