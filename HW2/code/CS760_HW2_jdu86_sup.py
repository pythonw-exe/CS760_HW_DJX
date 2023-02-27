# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.interpolate import lagrange
from collections import Counter
import matplotlib
from matplotlib import pyplot as plt
import scipy

def read_data(filename):
    file = open(filename, "r")
    lines = file.readlines()
    data = []
    for line in lines:
        data.append(line.split())
    data = np.array(data).astype(float)
    return data


def Q7_dataset(data):
    np.random.shuffle(data)
    d32 = data[0:31]
    d128 = data[0:127]
    d512 = data[0:511]
    d2048 = data[0:2047]
    d8192 = data[0:8191]
    dtest = data[8192:9999]
    return dtest, d32, d128, d512, d2048, d8192

#P2Q2
unsplittable_data = np.array([[1, 1, 1], [1, 5, 0], [5, 1, 0],[5, 5, 1]])
y0 = unsplittable_data[unsplittable_data[:, 2] == 0]
y1 = unsplittable_data[unsplittable_data[:, 2] == 1]
plt.scatter(y0[:, 0], y0[:, 1], color='g', label="y=0")
plt.scatter(y1[:,0], y1[:,1], color='r', marker='x', label="y=1")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")

#P3
data = read_data("Dbig.txt")
dtest, d32, d128, d512, d2048, d8192 = Q7_dataset(data)

clf = DecisionTreeClassifier(criterion="entropy")

model32 = clf.fit(d32[:,:2], d32[:,-1])
n32 = clf.tree_.node_count
error32 = 1-model32.score(dtest[:,:2], dtest[:,-1])

model128 = clf.fit(d128[:,:2], d128[:,-1])
n128 = clf.tree_.node_count
error128 = 1-model128.score(dtest[:,:2], dtest[:,-1])

model512 = clf.fit(d512[:,:2], d512[:,-1])
n512 = clf.tree_.node_count
error512 = 1-model512.score(dtest[:,:2], dtest[:,-1])

model2048 = clf.fit(d2048[:,:2], d2048[:,-1])
n2048 = clf.tree_.node_count
error2048 = 1-model2048.score(dtest[:,:2], dtest[:,-1])


model8192 = clf.fit(d8192[:,:2], d8192[:,-1])
n8192 = clf.tree_.node_count
error8192 = 1-model8192.score(dtest[:,:2], dtest[:,-1])

print("n",(n32,n128,n512,n2048, n8192),"error",(error32, error128, error512, error2048, error8192))

plt.figure()
plt.ylabel("err")
plt.xlabel("n")
plt.plot((32,128,512,2048, 8192),(error32, error128, error512, error2048, error8192))

#P4
a = 0
b = 2*np.pi

n = 100

X_train = np.linspace(a, b, n)
Y_train = np.sin(X_train)

X_s = np.random.choice(X_train, size = 20, replace = False)
Y_s = np.sin(X_s)

f = lagrange(X_s, Y_s)

X_test = np.linspace(a, b, n)
Y_test = np.sin(X_test)

train_error = np.mean((Y_train - f(X_train))**2)
test_error = np.mean((Y_test - f(X_test))**2)

print("Train error:", train_error)
print("Test error:", test_error)

epsilon_vals = [0.5, 1.0, 4.0]

for epsilon in epsilon_vals:
    X_train_noisy = X_train + np.random.normal(0, epsilon, n)
    Y_train_noisy = np.sin(X_train_noisy)

    X_s_noisy = np.random.choice(X_train_noisy, size = 20, replace = False)
    Y_s_noisy = np.sin(X_s_noisy)
    f = lagrange(X_s_noisy, Y_s_noisy)

    X_test_noisy = X_test + np.random.normal(0, epsilon, n)
    Y_test_noisy = np.sin(X_test_noisy)

    train_error = np.mean((Y_train_noisy - f(X_train_noisy))**2)
    test_error = np.mean((Y_test_noisy - f(X_test_noisy))**2)

    print("epsilon:", epsilon)
    print("Train error:", train_error)
    print("Test error:", test_error)
