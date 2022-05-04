from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    # classes表示不同类别的名称，比如这有6个类别
    classes=[]
    for i in range(54):
        if i%5==0:
           classes.append(i+1)
        else:
           classes.append("")
    classes.append('55')

    plt.figure(figsize=(9, 6), dpi=200)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('viridis'))
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
   # plt.xticks(xlocations, classes, rotation=90)
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    # plt.ylabel('Actual label')
    # plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    #plt.gca().set_xticks(tick_marks, minor=True)
    #plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

if __name__=='__main__':
    data_dir='C:/Users/huanghyuan/Desktop/ETRI Action/rgb.npz'
    # random_numbers = np.random.randint(55, size=500)  # 6个类别，随机生成50个样本
    # y_true = random_numbers.copy()  # 样本实际标签
    # random_numbers[:20] = np.random.randint(55, size=20)  # 将前10个样本的值进行随机更改
    # y_pred = random_numbers  # 样本预测标签
    content=np.load(data_dir,allow_pickle=True)
    y_true,y_pred=content['lable'],content['pred']


    # 获取混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cb = []
    for i in range(len(cm[0])):
        sum=0.0
        temp=[]
        for j in range(len(cm[0])):
            sum+=cm[i][j]
        for j in range(len(cm[0])):
            temp.append(cm[i][j]/sum)
        cb.append(temp)
    cb=np.array(cb)
    plot_confusion_matrix(cb, 'D:/code/Cross_MARS/ETRI/RGB/confusion_matrix.png', title='confusion matrix')
    print('finish')