from typing import List

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings

# Import packages to do the classifying
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import scipy as sp

gama = 1


def mahalanobis(x, m, inv_cov=None):
    x_minus_m = x - m
    left_term = np.dot(x_minus_m, inv_cov)
    mahal = np.dot(left_term, x_minus_m.T)
    return mahal


def euclidean(x, m):
    x_minus_m = x - m
    euclid = np.dot(x_minus_m, x_minus_m.T)
    return euclid


def cluster_on_mahalanobis_distance(x, means: List[dict] = None, inv_covs=None):
    mahal = []
    for mean in means:
        mahal.append(mahalanobis(x, list(mean.values())[0], inv_covs[list(mean.keys())[0]]))
        # mahal.append(mahalanobis(x, list(mean.values())[0]))
    mi_dist = mahal.index(min(mahal))
    return list(means[mi_dist].keys())[0]


def cluster_on_euclidean_distance(x, means: List[dict] = None):
    euclid = []
    for mean in means:
        euclid.append(euclidean(x, list(mean.values())[0]))
        # mahal.append(mahalanobis(x, list(mean.values())[0]))
    min_dist = euclid.index(min(euclid))
    return list(means[min_dist].keys())[0]


def get_distance_euclidean(x=None, y=None, x_test=None):
    dist = []
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(x, y, test_size=0.2)
    if x_test is not None:
        new_X_test = x_test
    dict = get_label_dict(new_X_train, new_y_train)
    c1 = get_means(dict[-1], -1)
    c2 = get_means(dict[1], 1)
    c1.extend(c2)
    means = c1
    for i in range(0, len(new_X_test)):
        a = cluster_on_euclidean_distance(new_X_test[i], means=means)
        dist.append(a)
    get_confusion_matrix(y_test=new_y_test,dist=dist)
    return dist


def get_confusion_matrix(y_test, dist):
    print('Confusion Matrix :')
    print(confusion_matrix(y_test, dist))
    print('Accuracy Score :', accuracy_score(y_test, dist))


def get_covariance_inverse(dict_part, part):
    inv_cov = sp.linalg.inv(np.cov(dict_part.T))
    return {part: inv_cov}


def get_distance(x=None, y=None, x_test=None):
    dist = []
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(x, y, test_size=0.2)
    if x_test is not None:
        new_X_test = x_test
    dict = get_label_dict(new_X_train, new_y_train)
    c1 = get_means(dict[-1], -1)
    c2 = get_means(dict[1], 1)
    c1.extend(c2)
    means = c1
    inv_cov1 = get_covariance_inverse(dict[-1], -1)
    inv_cov2 = get_covariance_inverse(dict[1], 1)
    inv_cov = {**inv_cov1, **inv_cov2}

    for i in range(0, len(new_X_test)):
        a = cluster_on_mahalanobis_distance(new_X_test[i], means=means, inv_covs=inv_cov)
        dist.append(a)
    get_confusion_matrix(y_test=new_y_test, dist=dist)
    return dist


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, test_idx=None, resolution=0.009, new_x=None):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    x_test = get_rbf_kernel(np.c_[xx1.ravel(), xx2.ravel()], gama)
    Z = get_distance_euclidean(new_X_xor,y_xor,x_test)
    print("hear")
    Z = np.asarray(Z, float)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


def generate_data():
    np.random.seed(0)
    X_xor = np.random.randn(2000, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                           X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    return X_xor, y_xor


def show_original_data(X_xor, y_xor):
    plt.scatter(X_xor[y_xor == 1, 0],
                X_xor[y_xor == 1, 1],
                c='b', marker='x',
                label='1')
    plt.scatter(X_xor[y_xor == -1, 0],
                X_xor[y_xor == -1, 1],
                c='r',
                marker='s',
                label='-1')

    plt.xlim(X_xor[:, 0].min() - 1, X_xor[:, 0].max() + 1)
    plt.ylim(X_xor[:, 1].min() - 1, X_xor[:, 1].max() + 1)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


X_xor, y_xor = generate_data()


# show_original_data(X_xor=X_xor, y_xor=y_xor)


def get_mean(dict_part, part):
    m = np.mean(dict_part, axis=0)
    return {part: m}


def get_label_dict(X_train, y_train) -> dict:
    return {label: X_train[y_train == label] for label in np.unique(y_train)}


def get_means(dict, part):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(dict)
    labels = kmeans.labels_
    new_dict = get_label_dict(dict, labels)
    means = []
    for key, value in new_dict.items():
        means.append(get_mean(value, part))
    return means


dict = get_label_dict(X_xor, y_xor)
m1 = get_means(dict[-1], -1)
m2 = get_means(dict[1], 1)
m1.extend(m2)

centers = m1


def get_rbf_kernel(X, gamma):
    print("gamma :", gamma)
    new_X = []

    for i, data in enumerate(X):
        new_data = []
        for j, center in enumerate(centers):
            d = np.exp(-sum(pow((np.subtract(data, list(center.values())[0])), 2)) / gamma)
            new_data.append(d)
        new_X.append(np.array(new_data))
    new_X = np.array(new_X)
    return new_X


new_X_xor = get_rbf_kernel(X_xor, gama)
# svm1 = SVC(kernel ='rbf',gamma=gama)
# svm1.fit(X_xor, y_xor)
show_original_data(X_xor, y_xor)
#
# plot_decision_regions(X_xor, y_xor, new_x=new_X_xor)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# get_distance(X_xor,y_xor)
# get_distance(X_xor,y_xor)
# get_distance(new_X_xor,y_xor)
get_distance_euclidean(new_X_xor,y_xor)
# get_distance_euclidean(X_xor,y_xor)