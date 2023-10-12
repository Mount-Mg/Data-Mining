import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#导入数据
data = pd.read_csv('D:\数据仓库\Wilt\Wilt\Wilt_withoutdupl_05.csv', header=0)
data.columns = ['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan','id','outlier'] #data['SD_pan']
x = data[['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan']]
y = data['outlier']
y = y.replace('no', 1)
y = y.replace('yes', -1)

from sklearn.neighbors import LocalOutlierFactor as lof
from sklearn.metrics import confusion_matrix
#LOF算法
lof_model = lof(n_neighbors=4, 
                contamination=0.2)  #contamination表示异常值所占数据集比例

y_pred = lof_model.fit_predict(x)
scores = lof_model.negative_outlier_factor_
print(scores)
matrix = confusion_matrix(y, y_pred)
print(matrix)   #混淆矩阵

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import seaborn as sns
fpr, tpr, thersholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)

#绘制LOF分数分布图
sns.distplot(-scores[y==1], label="inlier scores")
sns.distplot(-scores[y==-1], label="outlier scores").set_title("Distribution of Outlier Scores from LOF Detector")
plt.legend()
plt.xlabel("Outlier score")
#绘制ROC曲线图
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()