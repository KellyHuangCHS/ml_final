import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import svm, neighbors
from mpl_toolkits.mplot3d import Axes3D

def get_data():
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

    return X_train, y_train, X_test, y_test

def draw_points(X_train, y_train, X_test, y_test, v):
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

    # plot_points(X_train, y_train, c1='red', c2='blue')
    # plot_points(X_test, y_test, c1='yellow', c2='green')
    ax.scatter([s[0] for s in v], [s[1] for s in v], [s[2] for s in v], c='black', edgecolors='k', alpha=0.3)
    plt.show()

def write_csv(results, csv_name):
    save = pd.DataFrame(data=results)
    save.to_csv(csv_name, index=False)

def print_predict(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    v = []
    if(hasattr(clf, 'support_vectors_')):
        v = clf.support_vectors_
        # print(v)
    y_test_pred = clf.predict(X_test)
    num_correct = np.sum(y_test_pred == y_test)
    acc = float(num_correct) / len(y_test)
    print('Test data got %d / %d = %f accuracy' % (num_correct, len(y_test), acc))
    return v

def model_svm(X_train, y_train, X_test, y_test):
    def get_parameters():
        parameters = {'C':[], 'gamma': []}
        for i in range(1, 2):
            parameters['C'].append(i)
            a = float((i-1)/100)
            b = float(i/100)
            rand = (b - a)*np.random.sample() + a # 取[0.1, 0.2)之间的值, 以此类推
            parameters['gamma'].append(rand)
        return parameters

    # 交叉验证5次得出训练模型，选出最好的超参数对test进行训练
    best_score = 0.0
    best_parameters = {}
    scores = list()
    parameters = get_parameters()
    print(parameters)
    for gamma in parameters['gamma']:
        for C in parameters['C']:
            clf = svm.SVC(C=C, gamma=gamma)
            score = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
            this_score = np.mean(score)
            scores.append(this_score)
            print('gamma: %f, C: %f, Accuracy mean: %f' % (gamma, C, this_score,))
            if(this_score > best_score):
                best_parameters['gamma'] = gamma
                best_parameters['C'] = C
                best_score = this_score
    print(best_parameters)
    parameters['scores'] = scores
    write_csv(parameters, 'csv_result_svm.csv')
    clf = svm.SVC(C=best_parameters['C'], gamma=best_parameters['gamma']) # gamma分布在边界周围的取值范围在0.01-0.02之间
    # clf = svm.SVC(C=1, gamma=0.0383564)
    v = print_predict(clf, X_train, y_train, X_test, y_test)
    return v


def model_knn(X_train, y_train, X_test, y_test):
    param_K = np.arange(1,20)
    param_weights = ['distance', 'uniform']
    best_k = 1
    best_score = 0.0
    data = {'weights':[], 'K':[], 'Accuracy mean':[]}
    best_weights = ''
    for weights in param_weights:
        for k in param_K:
            clf = neighbors.KNeighborsClassifier(k, weights=weights)
            score = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
            this_score = np.mean(score)
            data['weights'].append(weights)
            data['K'].append(k)
            data['Accuracy mean'].append(this_score)
            print('weights: %s, K: %d, Accuracy mean: %f' % (weights, k, this_score,))
            if (this_score > best_score):
                best_weights = weights
                best_k = k
                best_score = this_score
    print(best_weights, best_k)
    write_csv(data, 'csv_result_knn.csv')
    clf = neighbors.KNeighborsClassifier(best_k, weights=best_weights)
    print_predict(clf, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data()
    # model_knn(X_train, y_train, X_test, y_test)
    v = model_svm(X_train, y_train, X_test, y_test)
    draw_points(X_train, y_train, X_test, y_test, v)