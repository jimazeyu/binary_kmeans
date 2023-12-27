import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")


# 中心点
X1,Y1=1,1
X2,Y2=5,3
X3,Y3=2,5
x1 = np.random.normal(X1, 1, 200)
y1 = np.random.normal(Y1, 1, 200)
x2 = np.random.normal(X2, 1, 200)
y2 = np.random.normal(Y2, 1, 200)
x3 = np.random.normal(X3, 1, 200)
y3 = np.random.normal(Y3, 1, 200)

dataset1 = [[x1[i],y1[i]] for i in range(200)]
dataset2 = [[x2[i],y2[i]] for i in range(200)]
dataset3 = [[x3[i],y3[i]] for i in range(200)]
dataset1 = np.array(dataset1)
dataset2 = np.array(dataset2)
dataset3 = np.array(dataset3)

dataset=np.append(dataset1,dataset2,axis=0)
dataset=np.append(dataset,dataset3,axis=0)

# 计算欧氏距离
def euclid(point1,point2,dimension):
    dis=0
    for i in range(dimension):
        dis+=(point1[i]-point2[i])**2
    return dis**0.5
# 更新centers
def update_centers(dataset,labels,clusters):
    length,dimension = dataset.shape
    centers=np.zeros([clusters,dimension])
    # 计算每个cluster的个数
    numbers=np.zeros(clusters)
    for label in labels:
        numbers[label]+=1
    for i,point in enumerate(dataset):
        for di in range(dimension):          
            centers[labels[i]][di]+=point[di]
    for i in range(clusters):
        # print(centers[i])
        # print(numbers[i])
        if(numbers[i]!=0):
            centers[i]/=numbers[i]
    return centers
# 可视化
def pplot(dataset,labels):  
    x = dataset[:,0]
    y = dataset[:,1]
    color_list=[ '#000080', '#006400','#00CED1', '#800000', '#800080',
                 '#CD5C5C', '#DAA520', '#E6E6FA', '#F08080', '#FFE4C4']
    cmap = [color_list[i] for i in labels]
    plt.scatter(x,y,c=cmap)

# 参数为数据集、簇个数、迭代次数
def k_means(dataset,clusters,iters,visualization=False,reduced_data=[]):
    plt.ion()
    # 样本个数
    length,dimension = dataset.shape
    # 随机生成初始簇心
    centers = dataset[random.sample(range(0,length),clusters)]
    #print(centers.shape) # clusters*dimension
    # 数据标签,代表每类所属于的簇
    labels = []
    for _ in range(iters):
        for center in centers:
            tmp_list = [euclid(dataset[i],center,dimension) for i in range(length)]
            labels.append(tmp_list)
        labels = np.argmin(labels,axis=0)
        centers = update_centers(dataset,labels,clusters)
        # 分布可视化
        if visualization:
            pplot(reduced_data,labels)
        plt.pause(1)
        if(_!=iters-1):
            labels = []
    plt.ioff()
    return labels

# 计算sse（簇内误差平方和）
def sse(dataset,labels,clusters):
    length,dimension = dataset.shape
    data = []
    if(clusters>1):
        for i in range(clusters):
            data.append([])
        for i,dt in enumerate(dataset):
            data[labels[i]].append(dt)
    else:
        data.append([])
        data[0] = dataset
    res=0
    for k in range(clusters):
        for point1 in data[k]:
            for point2 in data[k]:
                res+=euclid(point1,point2,dimension)**2
    return res
#sse(dataset,labels,3)

# 二分kmeans,iters代表每次尝试不同点进行二分的次数
def bi_kmeans(dataset,iters,clusters):    
    length = len(dataset)
    # 初始化标签
    labels=[]
    for i in range(length):
        labels.append(0)
    # 分割簇数-1次
    for it in range(clusters-1): 
        # 计算每类sse    
        data = []
        for i in range(it+1):
            data.append([])
        for i,dt in enumerate(dataset):
            data[labels[i]].append(dt) 
        max_sse=0
        tag=0
        for i in range(it+1):
            dt = np.array(data[i])
            tmp_sse = sse(dt,[],1)
            if tmp_sse>max_sse:
                max_sse=tmp_sse
                tag=i
        idx=[]
        for i in range(length):
            if labels[i]==tag:
                idx.append(i)
        dataset_to_divide = data[tag]
        dataset_to_divide = np.array(dataset_to_divide)
        f_labels = k_means(dataset_to_divide,2,1,visualization=False,reduced_data=dataset)
        f_sse = sse(dataset_to_divide,f_labels,2)
        for i in range(iters-1):
            tmp_labels = k_means(dataset_to_divide,2,1,visualization=False,reduced_data=dataset_to_divide)
            tmp_sse = sse(dataset_to_divide,tmp_labels,2)
            if tmp_sse<f_sse:
                f_sse=tmp_sse
                f_labels=tmp_labels
        for i,idxx in enumerate(idx):
            if f_labels[i]==0:
                labels[idxx]=it+1
        pplot(dataset,labels)
    return labels
     
labels=bi_kmeans(dataset,3,6) #每次二分迭代5次，最终分为8个簇

# 计算两个簇之间的平均距离
def average_distance(data1,data2):
    #print(data1.shape)
    _,dimension = data1.shape
    len1=len(data1)
    len2=len(data2)
    ssum=0;
    for i in range(len1):
        for j in range(len2):
            ssum+=euclid(data1[i],data2[j],dimension)
    #print(ssum,len1,len2)
    return ssum/(len1*len2)

# 层次聚类,每次将距离（使用平均距离）最小的两个簇合二为一
def hac(dataset,labels,clusters,target_clusters):
    if target_clusters>=clusters:
        return labels
    for it in range(clusters-target_clusters):
        # 找距离最短的两个簇   
        data = []
        for i in range(clusters-it):
            data.append([])
        for i,dt in enumerate(dataset):
            data[labels[i]].append(dt) 
        min_dis=1e10 #inf
        label1=0
        label2=0
        for i in range(clusters-it):
            for j in range(clusters-it):
                tmp_dis=average_distance(np.array(data[i]),np.array(data[j]))
                if(tmp_dis<min_dis and i!=j):
                    min_dis=tmp_dis
                    label1=i
                    label2=j
        label1,label2=max(label1,label2),min(label1,label2)
        #print(label1,label2)
        for i in range(len(labels)):
            if labels[i]==label1:
                labels[i]=label2
        for i in range(len(labels)):
            if labels[i]>label1:
                labels[i]-=1
        pplot(dataset,labels)
        plt.pause(1)
    return labels
tmp_labels=list(labels)
print("使用层次聚类合并")
_=hac(dataset,tmp_labels,6,3)