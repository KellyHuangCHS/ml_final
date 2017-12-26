import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

from classifiers.linear_classifier import LinearSVM

test_dir = './datasets/Testing-set-label.csv'
training_dir = './datasets/Training-set.csv'

training_data = pd.read_csv(training_dir)
test_data = pd.read_csv(test_dir)
X_train = np.array(training_data[["a","b","c"]])
y_train = np.array(training_data["t"])
X_test = np.array(test_data[["a","b","c"]])
y_test = np.array(test_data["t"])
# print(X_train, y_train)
# print(X_train.shape, y_train.shape)

def draw_points():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_points(X,y,c1,c2):
        one = X[np.argwhere(y==1)]
        zero = X[np.argwhere(y==0)]
        ax.scatter([s[0][0] for s in one], [s[0][1] for s in one], [s[0][2] for s in one], c=c1, edgecolors='k', alpha=0.3)
        ax.scatter([s[0][0] for s in zero], [s[0][1] for s in zero], [s[0][2] for s in zero], c=c2, edgecolors='k', alpha=0.3)
        ax.set_xlabel('Feature a')
        ax.set_ylabel('Feature b')
        ax.set_zlabel('Feature c')

    plot_points(X_train, y_train, c1='red', c2='blue')
    plot_points(X_test, y_test, c1='yellow', c2='green')
    plt.show()

# draw_points()

# 自带的grid_search，kfold=5
# parameters = {'kernel':('poly', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(X_train, y_train)
# y_test_pred = clf.predict(y_test)
# num_correct = np.sum(y_test_pred == y_test)
# acc = float(num_correct)/len(y_test)
# print('Test data got %d / %d = %f accuracy'%(num_correct, len(y_test), acc))

# 交叉验证5次得出训练模型，选出最好的超参数对test进行训练
C_choice = [1.0, 2.0, 3.0, 4.0, 5.0]
gamma_choice = [0.125, 0.25, 1, 2, 4]
kernel_chioce = ['poly', 'rbf']
C_to_scores = {}
best_C = 0.0
best_gamma = 0.0
best_kernel = ''
best_score = 0.0
for kernel in kernel_chioce:
    for gamma in gamma_choice:
        for C in C_choice:
            clf = svm.SVC(C=C)
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
            this_scores = np.mean(scores)
            C_to_scores[C] = this_scores
            print('kernel: %s, gamma: %f, C: %f Accuracy mean: %f' % (kernel, gamma, C, this_scores,))
            if(this_scores > best_score):
                best_C = C
                best_gamma = gamma
                best_kernel = kernel
                best_score = this_scores

print(best_C, best_kernel, best_gamma)
clf = svm.SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
num_correct = np.sum(y_test_pred == y_test)
acc = float(num_correct)/len(y_test)
print('Test data got %d / %d = %f accuracy'%(num_correct, len(y_test), acc))

# 画一个grid_search图片


