# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

def read_txt(filename):
    x = []
    y = [] 
    with open(filename, 'r') as data:
        lines = data.readlines()
        for line in lines:
            x.append([float(line.split()[0]), float(line.split()[1])])
            y.append(int(line.split()[2]))
    return x, y

def read_email(filename):
    y = []
    x = []
    with open(filename,'r') as f:
        for line in f:
            if 'Prediction' in line:
                continue
            line = line.split(',')[1:]
            y.append(int(line[-1]))
            line = line[:-1]
            current_x = []
            for item in line:
                current_x.append(int(item))
            x.append(current_x)
    return np.array(x),np.array(y)

#P1Q5
plt.figure()
plt.plot([0, 0,1/4,1/4,2/4,2/4,1], [0, 1/3, 1/3, 2/3, 2/3, 1, 1])

plt.grid('on')
plt.xlabel('FP rate')
plt.ylabel('TP rate')
plt.title('ROC curve')

##P2
#Q1
data, result = read_txt('./D2z.txt')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data, result)

test_data = np.mgrid[-2:2.1:0.1, -2:2.1:0.1].reshape(2,-1).T
y_pred = knn.predict(test_data)

plt.figure()
plt.scatter(test_data[y_pred==0, 0], test_data[y_pred==0, 1], c='r', s=1, label='Class: 0')
plt.scatter(test_data[y_pred==1, 0], test_data[y_pred==1, 1], c='b', s=1, label='Class: 1')
plt.scatter(np.array(data)[np.array(result)==0,0], np.array(data)[np.array(result)==0,1], c='g', marker='+', label='training class: 0')
plt.scatter(np.array(data)[np.array(result)==1,0], np.array(data)[np.array(result)==1,1], c='k', marker='x', label='training class: 1')

plt.xlim([-2, 2])
plt.ylim([-2, 2])

#Q2&Q4
data, result = read_email('./emails.csv')

acc_list = []
precision_list = []
recall_list = []

for i in range(5):
    test_data = data[:1000, :]
    train_data = data[1000:, :]
    test_result = result[:1000]
    train_result = result[1000:] 
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, train_result)
    test_pred = knn.predict(test_data)
    
    acc = np.sum(test_result == test_pred) / len(test_result)
    precision = np.sum(test_pred[test_result==1]) / np.sum(test_pred==1)
    recall = np.sum(test_pred[test_result==1]) / np.sum(test_result==1)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)

    print(i, "accuracy", acc)
    print(i, "precision", precision)
    print(i, "recall", recall)
    
    data = np.roll(data, -1000, axis=0)
    result = np.roll(result, -1000)


data, result = read_email('./emails.csv')

acc_k = []
precision_k = []
recall_k = []

for k in [1, 3, 5, 7, 10]:
    acc_list = []
    precision_list = []
    recall_list = []
    for ii in range(5):
        test_data = data[:1000, :]
        train_data = data[1000:, :]
        test_result = result[:1000]
        train_result = result[1000:] 
    
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, train_result)
        test_pred = knn.predict(test_data)
        data = np.roll(data, -1000, axis=0)
        result = np.roll(result, -1000)

        acc = np.sum(test_result == test_pred) / len(test_result)
        precision = np.sum(test_pred[test_result==1]) / np.sum(test_pred==1)
        recall = np.sum(test_pred[test_result==1]) / np.sum(test_result==1)

        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        
        data = np.roll(data, -1000, axis=0)
        result = np.roll(result, -1000)
    
    acc_avg = np.mean(acc_list)
    precision_avg = np.mean(precision_list)
    recall_avg = np.mean(recall_list)

    acc_k.append(acc_avg)
    precision_k.append(precision_avg)
    recall_k.append(recall_avg)

    print(k, "avg accuracy", acc_avg)
    #print(k, "avg precision", precision_avg)
    #print(k, "avg recall", recall_avg)

plt.figure()
plt.plot([1,3,5,7,10], acc_k, '-o')

plt.grid('on')
plt.xlabel('k')
plt.ylabel('Average accuracy')
plt.title('KNN 5-fold cross validation')

#Q3
# class LogisticRegression:
#     def __init__(self, learning_rate=0.1, max_iter=50):
#         self.learning_rate = learning_rate
#         self.num_iterations = max_iter
    
#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))
    
#     def fit(self, X, y):
#         m, n = X.shape
#         self.theta = np.zeros((n, 1))
        
#         for i in range(self.num_iterations):
#             z = np.dot(X, self.theta)
#             h = self.sigmoid(z)
#             gradient = np.mean(np.dot(X.T, (h - y)) / m,1).reshape(-1,1)
#             self.theta -= self.learning_rate * gradient
    
#     def predict(self, X):
#         return np.round(self.sigmoid(np.dot(X, self.theta)))

#     def predict_proba(self, X):
#         return self.sigmoid(np.dot(X, self.theta))

data, result = read_email('./emails.csv')

acc_r = []
precision_r = []
recall_r = []

for i in range(5):
    test_data = data[:1000, :]
    train_data = data[1000:, :]
    test_result = result[:1000]
    train_result = result[1000:] 
    
    #log = LogisticRegression()
    log = LogisticRegression(solver='liblinear', max_iter = 15)
    log.fit(train_data, train_result)

    test_pred = log.predict(test_data)

    acc = np.sum(test_result == test_pred) / len(test_result)
    precision = np.sum(test_pred[test_result==1]) / np.sum(test_pred==1)
    recall = np.sum(test_pred[test_result==1]) / np.sum(test_result==1)

    acc_r.append(acc)
    precision_r.append(precision)
    recall_r.append(recall)

    print(i, "accuracy", acc)
    print(i, "precision", precision)
    print(i, "recall", recall)
    
    data = np.roll(data, -1000, axis=0)
    result = np.roll(result, -1000)

#Q5
data, result = read_email('./emails.csv')
test_data = data[:1000, :]
train_data = data[1000:, :]
test_result = result[:1000]
train_result = result[1000:]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data, train_result)
test_pred_knn = knn.predict_proba(test_data)
fpr_knn, tpr_knn, _ = roc_curve(test_result, test_pred_knn[:,1])
auc_knn = auc(fpr_knn,tpr_knn)

#log = LogisticRegression()
log = LogisticRegression(solver='liblinear', max_iter=5)
log.fit(train_data, train_result)

test_pred_log = log.predict_proba(test_data)
#fpr_log, tpr_log, _ = roc_curve(test_result, test_pred_log)
fpr_log, tpr_log, _ = roc_curve(test_result, test_pred_log[:,1])
auc_log = auc(fpr_log, tpr_log)

plt.figure()
plt.plot(fpr_knn, tpr_knn,'b',label='KNeighbors Classifier (AUC=%.2f)' % auc_knn)
plt.plot(fpr_log, tpr_log,'r',label='Logistic Regression (AUC=%.2f)' % auc_log)
plt.title('ROC Curve')

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('FP rate')
plt.ylabel('TP rate')
plt.legend()
