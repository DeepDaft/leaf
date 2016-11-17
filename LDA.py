from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics


import numpy as np


def use_LogR(X_train, X_test, y_train, y_test):


    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    LogisticRegression(penalty='L2',max_iter=2)
    score = lr.score(X_test,y_test)
    return (score)

def use_LDA(X_train, X_test, y_train, y_test):

    # 0.0 acc
    # clf = SVC()
    # clf.fit(X_train,y_train)
    # # SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    # #     decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
    # #     max_iter=-1, probability=True, random_state=None, shrinking=True,
    # #     tol=0.001, verbose=False)
    # SVC()
    #
    # score = clf.score(X_test,y_test)

    from sklearn.svm import LinearSVC
    clf = LinearSVC()

    clf.fit(X_train, y_train)
    LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True,
              intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
    score = clf.score(X_test, y_test)
    print("score of SVM is: "+ repr(score))
    # return clf._predict_proba_lr(x_test),clf.classes_
    from sklearn.lda import LDA
    clf = LDA()
    clf.fit(X_train, y_train)
    LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
        store_covariance=False, tol=0.0001)
    score = clf.score(X_test,y_test)
    print("score of lda is: "+ repr(score))









# load files
def load_data(filename,flag):
    f = open(filename,'r')
    x = []
    y = []
    data = f.readlines()
    if flag:
        for line in data[200:]:

            seq = line.split(',')
            if seq[0] == 'id':
                continue
            features = [float(n) for n in seq[2:]]
            x.append(features)
            y.append(seq[1].strip())
    else:
        for line in data[:200]:
            seq = line.split(',')
            if seq[0] == 'id':
                continue
            features = [float(n) for n in seq[2:]]
            x.append(features)
            y.append(seq[1].strip())
            # for line in f.readlines():
    #     seq = line.split(',')
    #     if seq[0] == 'id':
    #         continue
    #     if flag:
    #         features = [float(n) for n in seq[2:]]
    #         x.append(features)
    #         y.append(seq[1].strip())
    #     else:
    #         features = [float(n) for n in seq[1:]]
    #         x.append(features)
    #         y.append(seq[1].strip())


    return x, y

if __name__ == '__main__':

    x_train, y_train = load_data('/Users/wuhao/Downloads/train.csv', True)

    x_test, y_test = load_data('/Users/wuhao/Downloads/train.csv', False)

    n_values = len(set(y_train))
    print (np.shape(y_train), n_values)

    # print (np.max(y_train),np.min(y_train))

    for index, item in enumerate(set(y_train)):
        # print (item)

        if index % 10 == 0:
            print('\n')

    use_LDA(x_train, x_test, y_train, y_test)
    #accuracy 95.9%

    # print result
    # test for y
    # m = np.eye(n_values)[y_train]
