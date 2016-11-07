

import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

import matplotlib.pyplot as plt


def get_train_test(the_set, split_size = 0.8):
    """
    using this function to split a pandas dataframe into
    training set and test set by giving the split size
    which also represents training set size.
    :param set:
    :param split_size: double, default 0.8
    :return:
        this function will return two values,
        first is the train set
        next is the test set
        both of the will be pandas dataframe
    """
    msk = np.random.rand(len(the_set)) < split_size
    return the_set[msk], the_set[~msk]


def get_x_y(the_set):
    """
    this function are used to get x (attributes) and y (species)
    for a given set. this set should be a pandas dataframe which
    will contains a column called 'species'.

    in this process, we will numeric each species as y then drop the
    species return as x, for next step training and verifying.
    :param set:
    :return:
    x, leaf attributes, data type numpy matrix
    y, leaf species
    """

    y = the_set.species
    species = y.unique().tolist()
    y = [species.index(i) for i in y]

    the_set = the_set.drop('species', 1)
    x = the_set.as_matrix()
    return x, y


def pca_selection(train_x, n_components=None):
    """
    pca feature selection
    according to the graph that shows the variance of each attributes
    then choose how many attributes to maintain manually.

    :param train_x:
    :param n_components:
    :return:
    """
    pca_estimator = PCA(n_components=n_components).fit(train_x)

    plt.plot(pca_estimator.explained_variance_ , linewidth=2)
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()

    return pca_estimator.transform(train_x)


def extratree_selection(train_x, train_y):
    """
    extratree feature selection
    a variant of a random forest
    blow the link that explain it with random forest
    https://www.quora.com/What-is-extra-trees-algorithm-in-machine-learning
    http://stackoverflow.com/questions/22409855/randomforestclassifier-vs-extratreesclassifier-in-scikit-learn

    :param train_x:
    :param train_y:
    :return:
    """
    clf_estimator = ExtraTreesClassifier().fit(train_x, train_y)
    select_estimator = SelectFromModel(clf_estimator, prefit=True)
    return select_estimator.transform(train_x)


def linearsvc_selection(train_x, train_y):
    """
    linear svm feature selection


    :param train_x:
    :param train_y:
    :return:
    """
    lsvc_estimator = LinearSVC(C=0.5, penalty='l1', dual=False).fit(train_x, train_y)
    select_estimator = SelectFromModel(lsvc_estimator, prefit=True)
    return select_estimator.transform(train_x)


def kbest_selection(train_x, train_y, k):
    """
    a kind of a wrapper

    :param train_x:
    :param train_y:
    :param k:
    :return:
    """
    return SelectKBest(chi2, k=k).fit_transform(train_x, train_y)


def feature_selection(train_x, train_y=None, solver='pca', n_components=None, k=8):
    if solver == 'pca':
        return pca_selection(train_x, n_components)
    elif solver == 'extraTree' and train_y is not None:
        return extratree_selection(train_x, train_y)
    elif solver == 'linearSVC' and train_y is not None:
        return linearsvc_selection(train_x, train_y)
    elif solver == 'kbest' and train_y is not None:
        return kbest_selection(train_x, train_y, k)
    elif train_y is None:
        print 'train_y is none type'
    else:
        print 'wrong solver type, we only take pca, extraTree, linearSVC and kBest'


def logistic_model(train_x, train_y, test_x, test_y):
    """
    when all the set to be ready, using logistic model to train
    then test.

    note: if the solver is liblinear, the max_iter will only control it
    if the setting max_iter lower than it automatic max_iter.
    for example, if the max_iter equals to 2, the max iteration of
    logistic regression will be 2. however, if we set max_iter as 200 (default 100)
    the max iteration for our kaggle (leaf classify) is 8 (according to dataset).

    question is not sure about the convergence function LogisticRegression use.
    it use the same function as svm use.
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    score, mean accuracy among all test data
    """
    # liblinear
    log_regre = lm.LogisticRegression(n_jobs=-1, solver='liblinear', max_iter=100)

    # multi class using lbfgs
    # log_regre = lm.LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100,
    #                                 multi_class='multinomial', n_jobs=8)

    # multi class using newton-cg
    # log_regre = lm.LogisticRegression(penalty='l2', solver='newton-cg', max_iter=5000,
    #                                 multi_class='multinomial', n_jobs=8)

    log_regre.fit(train_x, train_y)
    return log_regre.score(test_x, test_y)


def logistic_model_sgd(train_x, train_y, test_x, test_y):
    """
    to verify the iteration number of the above function.
    we manually using stochastic gradient descent instead of
    the convergence function in LogisticRegression.

    the results of logistic_svd almost the same as previous one.
    however, if we test it on training data itself, we found
    the accuracy on training set fit well, which can be
    regarded as over-fitting.

    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    score, mean accuracy among all test data
    """
    sgd = SGDClassifier(loss='log', penalty='l2', n_iter=1000,
                        n_jobs=-1)
    sgd.fit(train_x, train_y)
    return sgd.score(test_x, test_y)


def main():
    df_train = pd.DataFrame.from_csv("/home/troy/Kaggle/Leaf Classification/data/train.csv")
    df_test = pd.DataFrame.from_csv("/home/troy/Kaggle/Leaf Classification/data/test.csv")

    x, y = get_x_y(df_train)
    x = feature_selection(x, y, solver='pca', n_components=5, k=10)
    
    # check how many feature select
    print x.shape
    x = pd.DataFrame(x)
    x['species'] = pd.Series(y, index=x.index)

    train, test = get_train_test(x)
    train_x, train_y = get_x_y(train)
    test_x, test_y = get_x_y(test)

    score = logistic_model(train_x, train_y, test_x, test_y)
    print score

    score = logistic_model_sgd(train_x, train_y, test_x, test_y)
    print score


if __name__ == '__main__':
    main()



