# ESE-FN

## 环境
pytorch,python,numpy

## 数据准备
1. 首先调用util.py ETRI_skeleton_train_val 方法划分训练和测试集
2. 调用data_gen/gen_etridata.py 方法生成骨骼点数据。

## 分阶段训练
1. 调用 skeleton_train.py 训练骨骼骨干网络
2. 调用 rgb_train.py 训练RGB骨干网络
3. 调用 two_stream_fusion.py 训练融合网络

## 结果
![image](https://user-images.githubusercontent.com/34670345/166742379-e75e337c-f013-4cfa-b576-5da5e17162d9.png)

![image](https://user-images.githubusercontent.com/34670345/166742475-4aa7158e-bb92-449a-a9d2-f7e0758e504f.png)
