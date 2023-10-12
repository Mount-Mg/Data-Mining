import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#导入数据
data = pd.read_csv('D:\数据仓库\Wilt\Wilt\Wilt_withoutdupl_05.csv', header=0)
data.columns = ['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan','id','outlier'] #data['SD_pan']

#决策树算法
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
from six import StringIO
#划分训练集和测试集
x = data[['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan']]
y = data['outlier']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)
#决策树模型
tree_model = tree.DecisionTreeClassifier( criterion='gini', 
                                          splitter='best',
                                          max_depth=None,
                                          min_samples_leaf=3,
                                          ccp_alpha=0.0,
                                          )
tree_model = tree_model.fit(x_train, y_train)
#可视化决策树
dot_data = StringIO()
feature_names = ['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan']
class_names = ['yes', 'no']
dot_tree = tree.export_graphviz(tree_model,						    #模型
                                feature_names=feature_names,		#特征名
                                class_names=class_names,		    #类名
                                filled=True,						#是否填充
                                rounded=True,						#是否圆角
								out_file=dot_data,					#输出文件名
                                special_characters=True				#特殊字符
                                )

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("DicisionTree.pdf")
#评估决策树算法
result_train = tree_model.score(x_train, y_train)
result_test = tree_model.score(x_test, y_test)
print('result_train:', result_train,        #训练集准确率
      '\n', 'result_test:', result_test)    #测试集准确率
print([*zip(feature_names, tree_model.feature_importances_)])   #各属性的重要性指标
y_pred = tree_model.predict(x_test)
matrix = confusion_matrix(y_test, y_pred)   #混淆矩阵
print(matrix)         

from sklearn.metrics import roc_curve, auc
y_test = y_test.replace('yes',-1)
y_test = y_test.replace('no',1)
y_pred[y_pred=='yes'] = -1
y_pred[y_pred=='no'] = 1
fpr, tpr, thersholds = roc_curve(y_test, y_pred)
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
















               


