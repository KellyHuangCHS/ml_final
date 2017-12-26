import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

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
parameters = {'kernel':('poly', 'rbf'), 'C':[1, 2, 3, 4, 5], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
# print(parameters['kernel'], parameters['C'], parameters['gamma'])
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(X_train, y_train)
#
# cv_result = pd.DataFrame.from_dict(clf.cv_results_)
# with open('cv_result.csv', 'w') as f:
#     cv_result.to_csv(f)
# print("Best parameters: ")
# print(clf.best_params_)
#
# y_test_pred = clf.predict(y_test)
# num_correct = np.sum(y_test_pred == y_test)
# acc = float(num_correct)/len(y_test)
# print('Test data got %d / %d = %f accuracy'%(num_correct, len(y_test), acc))

# 交叉验证5次得出训练模型，选出最好的超参数对test进行训练
best_score = 0.0
best_parameters = {}
scores = list()
for kernel in parameters['kernel']:
    for gamma in parameters['gamma']:
        for C in parameters['C']:
            clf = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            score = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
            this_score = np.mean(score)
            scores.append(this_score)
            print('kernel: %s, gamma: %f, C: %f Accuracy mean: %f' % (kernel, gamma, C, this_score,))
            if(this_score > best_score):
                best_parameters['kernel'] = kernel
                best_parameters['gamma'] = gamma
                best_parameters['C'] = C
                best_score = this_score

print(best_parameters)
clf = svm.SVC(C=best_parameters['C'], kernel=best_parameters['kernel'], gamma=best_parameters['gamma'])
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
num_correct = np.sum(y_test_pred == y_test)
acc = float(num_correct)/len(y_test)
print('Test data got %d / %d = %f accuracy'%(num_correct, len(y_test), acc))

# 画一个grid_search图片


