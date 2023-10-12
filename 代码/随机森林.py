import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#导入数据
data = pd.read_csv('D:\数据仓库\Wilt\Wilt\Wilt_withoutdupl_05.csv', header=0)
data.columns = ['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan','id','outlier'] #data['SD_pan']

#随机森林算法
from sklearn.model_selection import train_test_split
#划分训练集和测试集
x = data[['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan']]
y = data['outlier']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y)
y_train = y_train.replace('yes',-1)
y_test = y_test.replace('yes',-1)
y_train = y_train.replace('no',1)
y_test = y_test.replace('no',1)
#随机森林模型
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100,
                                max_depth=None,
                                min_samples_leaf=3,
                                class_weight="balanced")
forest = forest.fit(x_train, y_train)
y_train_pred = forest.predict(x_train)
y_test_pred = forest.predict(x_test)
result_train = forest.score(x_train, y_train)
result_test = forest.score(x_test, y_test)
print('result_train:', result_train,        #训练集准确率
      '\n', 'result_test:', result_test)    #测试集准确率
feature_names = ['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan']
print([*zip(feature_names, forest.feature_importances_)])   #各属性的重要性指标
#评估模型效果
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
matrix = confusion_matrix(y_test, y_test_pred)
print(matrix)   #混淆矩阵
y_test = y_test.replace('yes',-1)
y_test = y_test.replace('no',1)
y_test_pred[y_test_pred=='yes'] = -1
y_test_pred[y_test_pred=='no'] = 1
fpr, tpr, thersholds = roc_curve(y_test, y_test_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()