import pandas as pd
import matplotlib.pyplot as plt

#导入数据
data1 = pd.read_csv('D:\数据仓库\Wilt\Wilt\Wilt_norm_05.csv', header=0)
data2 = pd.read_csv('D:\数据仓库\Wilt\Wilt\Wilt_withoutdupl_05.csv', header=0)
data3 = pd.read_csv('D:\数据仓库\Wilt\Wilt\Wilt_PCA.csv', header=None)
data3.columns = ['1','2','3']
#data2.columns = ['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan','id','outlier']
#x = data2[['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan']]
x = data3[['1','2']]
y = data1['outlier']
y = y.replace('no', 0)
y = y.replace('yes', -1)

#DBScan算法
from sklearn.cluster import DBSCAN
db_model = DBSCAN(eps=0.004,                    #邻域半径
                  min_samples=3,            #最小样本点数
                  metric='minkowski',       #最近邻距离度量参数
                  metric_params=None,               
                  algorithm='auto',         #最近邻搜索算法参数
                  leaf_size=30,             #停止建立子树的叶子节点数量的阈值
                  p=5).fit(x)
labels = db_model.labels_
print('噪声个数:', len(labels[labels[:] == -1]))
print('噪声比:', format(len(labels[labels[:] == -1]) / len(labels), '.2%'))
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('簇的个数：', n_clusters_)

#效果评估
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
for i in set(labels):
    cnt = 0
    for j in range(len(labels)):
        if labels[j] == i and y[j] == -1:
            cnt = cnt+1
    print(i, ":", len(labels[labels[:] == i]), " ", cnt)
#ROC,auc
y = y.replace(0,1)
fpr, tpr, thersholds = roc_curve(y, labels)
roc_auc = auc(fpr, tpr)
print('auc:', roc_auc)
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
#可视化结果
plt.scatter(x['1'], x['2'], s=1, c=labels)
plt.show()
#混淆矩阵
labels[labels != -1] = 1
matrix = confusion_matrix(y, labels)
print('混淆矩阵:\n', matrix)