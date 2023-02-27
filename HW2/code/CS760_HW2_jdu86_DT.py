# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

import matplotlib
from matplotlib import pyplot as plt
import scipy

#parameters
HW2Q3 = False

class Tree():
    def __init__(self):
        self.node = None
        self.data = None
        self.depth = None
        self.predict = None
        self.left = None
        self.right = None
        self.x1_low = None
        self.x1_up = None
        self.x2_low = None
        self.x2_up = None
        
    
def split_data(data, split):
    global data_dim
    dim = int(split[1])
    threshold = split[0]
    #print("split_data", dim, type(dim), threshold)
    data_dim = data[:, dim]
    data_left= data[data_dim<threshold]
    data_right = data[data_dim>=threshold]
    #print("split_data left", data_left, "split_data right", data_right)
    return data_left, data_right
    
def cal_entropy(data):
    y_data = data[:,2]   
    if (len(data[y_data == 0])+len(data[y_data == 1])) == 0:
        entropy = 0
        return entropy
    
    p_negative = len(data[y_data == 0])/(len(data[y_data == 0])+len(data[y_data == 1]))
    p_positive = len(data[y_data == 1])/(len(data[y_data == 0])+len(data[y_data == 1]))
    if p_negative == 0:
        entropy = -p_positive*np.log2(p_positive)
    elif p_positive == 0:
        entropy = -p_negative*np.log2(p_negative)
    else:
        entropy = -p_negative*np.log2(p_negative)-p_positive*np.log2(p_positive)
    #print("cal_entropy", entropy)
    return entropy

def cal_split_entropy(data_left, data_right):
    p_left= len(data_left)/(len(data_left)+len(data_right))
    p_right = len(data_right)/(len(data_left)+len(data_right))    
    if p_left == 0 and p_right == 0:
        entropy = 0
    elif p_left == 0:
        entropy = -p_right*np.log2(p_right)
    elif p_right == 0:
        entropy = -p_left*np.log2(p_left)
    else:
        entropy = -p_left*np.log2(p_left)-p_right*np.log2(p_right)
    return entropy

def cal_cond_entropy(data_left, data_right):
    p_left= len(data_left)/(len(data_left)+len(data_right))
    p_right = len(data_right)/(len(data_left)+len(data_right))
    entropy = p_left*cal_entropy(data_left)+p_right*cal_entropy(data_right)
    #print("cod_entropy", entropy, p_left, p_right)
    return entropy
    
def cal_info_gain_ratio(data, data_left, data_right):
    overall_entropy = cal_entropy(data)
    cond_entropy = cal_cond_entropy(data_left, data_right)
    split_entropy = cal_split_entropy(data_left, data_right)
    if split_entropy != 0:
        info_gain_ratio = (overall_entropy - cond_entropy)/split_entropy
    else:
        if HW2Q3 == True:
            info_gain = overall_entropy - cond_entropy
            print("[cal_info_gain_ratio] info gain ratio zero, info gain instead", info_gain)
        info_gain_ratio = 0
        
    #print("cal_info", info_gain)
    return info_gain_ratio
    
def find_best_split(data, split_list):
    best_info_gain_ratio = 0
    best_split = []
    if len(data) == 0:
        return []
    for i in split_list:
        #print("find_best", i)
        left_data, right_data = split_data(data, i)
        if len(left_data) == 0 and len(right_data) == 0:
            continue
        info_gain_ratio = cal_info_gain_ratio(data, left_data, right_data)
        
        if HW2Q3 == True:
            print("[find_best_split]", "feature", "x"+str(int(i[1])+1), "  thresold", i[0], "  info_gain_ratio", info_gain_ratio)
        
        if info_gain_ratio >= best_info_gain_ratio:
            best_info_gain_ratio = info_gain_ratio
            best_split = i
    
    if best_info_gain_ratio == 0:
        return []
    #print("find_best----------------")
    return best_split
    
def determine_candidate_splits(data):
    split_list = []
    #print("determin",data)
    #print("determin",data[:,0])
    x1_split = np.unique(data[:,0]).reshape(-1,1)
    x1_list = np.concatenate((x1_split, np.zeros(len(x1_split),int).reshape(-1,1)),axis=1)
    x2_split = np.unique(data[:,1]).reshape(-1,1)
    x2_list = np.concatenate((x2_split, np.ones(len(x2_split),int).reshape(-1,1)),axis=1)
    #print("determine",x1_list, x2_list)
    split_list = np.concatenate((x1_list,x2_list))
    #print(split_list)
    return split_list
    
def make_sub_tree(data, tree, depth):
    #print("sub_tree", data)
    split_list = determine_candidate_splits(data)
    best_split = find_best_split(data, split_list)
    tree.node = best_split
    tree.data = data
    tree.depth = depth
    if HW2Q3 == True:
        print("[make_sub_tree]", "depth", tree.depth)
    #print("here1",tree.node)
    if len(best_split) == 0:
        #print("leaf node",best_split, tree.node)
        tree.predict = predict(data)
        return tree
    #print("---------------------")
    data_left, data_right = split_data(data, best_split)
    #print("sub_tree, left", data_left)
    left_tree = Tree()
    right_tree = Tree()
    depth = depth + 1
    tree.left = make_sub_tree(data_left, left_tree, depth)
    tree.right= make_sub_tree(data_right, right_tree, depth)
    
    return tree
    
def predict(data):
    most_commmons = int(Counter(data[:,2]).most_common(1)[0][0])
    return most_commmons
    
def print_tree(tree):
    spaces = " "*tree.depth*4
    if tree.node != []:
        print(spaces, "-node-", "feature:", "x"+str(int(tree.node[1])+1), "cut:", tree.node[0])
    else:
        print(spaces, "-leaf-", "prediction:", tree.predict)

    if tree.left != None:
        print(spaces,"-left_subtree-")
        print_tree(tree.left)
        
    if tree.right != None:
        print(spaces,"-right_subtree-")
        print_tree(tree.right)
    
def read_data(filename):
    file = open(filename, "r")
    lines = file.readlines()
    data = []
    for line in lines:
        data.append(line.split())
    data = np.array(data).astype(float)
    return data

def plot_boundary(data, tree):
    fig = plt.figure()
    y_data = data[:,2]
    positive_data = data[y_data == 1]
    negative_data = data[y_data == 0]
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(positive_data[:,0], positive_data[:,1],  c = 'blue', marker='o', label = "1")
    plt.scatter(negative_data[:,0], negative_data[:,1],  c = 'red', marker = 'x', label = "0")
    plot_tree(tree)
    plt.legend()

def make_sub_tree_plot(data, tree, depth, x1_low, x1_up, x2_low, x2_up):
    #print("sub_tree", data)
    tree.x1_low = x1_low
    tree.x1_up = x1_up
    tree.x2_low = x2_low
    tree.x2_up = x2_up
    split_list = determine_candidate_splits(data)
    best_split = find_best_split(data, split_list)
    tree.node = best_split
    tree.data = data
    tree.depth = depth
    if HW2Q3 == True:
        print("[make_sub_tree]", "depth", tree.depth)
    #print("here1",tree.node)
    if len(best_split) == 0:
        #print("leaf node",best_split, tree.node)
        tree.predict = predict(data)
        return tree
    #print("---------------------")
    data_left, data_right = split_data(data, best_split)
    #print("sub_tree, left", data_left)
    left_tree = Tree()
    right_tree = Tree()
    depth = depth + 1
    if int(best_split[1] == 0):
        tree.left = make_sub_tree_plot(data_left, left_tree, depth, x1_low, best_split[0], x2_low, x2_up)
        tree.right= make_sub_tree_plot(data_right, right_tree, depth, best_split[0], x1_up, x2_low, x2_up)
    elif int(best_split[1] == 1):
        tree.left = make_sub_tree_plot(data_left, left_tree, depth, x1_low, x1_up, x2_low, best_split[0])
        tree.right = make_sub_tree_plot(data_right, right_tree, depth, x1_low, x1_up, best_split[0], x2_up)
    
    return tree
    
def plot_tree(tree):
    if tree.node != []:
        if int(tree.node[1]) == 0:
            plt.plot((tree.node[0],tree.node[0]),(tree.x2_low, tree.x2_up))
        elif int(tree.node[1]) == 1:
            plt.plot((tree.x1_low, tree.x1_up),(tree.node[0],tree.node[0]))
    else:
        if tree.predict == 1:
            plt.fill((tree.x1_low,tree.x1_low, tree.x1_up, tree.x1_up), (tree.x2_low, tree.x2_up, tree.x2_up,tree.x2_low), "b", alpha=0.2)
        elif tree.predict == 0:
            plt.fill((tree.x1_low,tree.x1_low, tree.x1_up, tree.x1_up), (tree.x2_low, tree.x2_up, tree.x2_up, tree.x2_low), "r", alpha=0.2)
        
    if tree.left != None:
        plot_tree(tree.left)
        
    if tree.right != None:
        plot_tree(tree.right)
        
def find_boundary(tree, boundary):
    if tree.node == []:
        boundary.append((tree.x1_low, tree.x1_up, tree.x2_low, tree.x2_up, tree.predict))
        return boundary
    
    if tree.left != None:
        boundary = find_boundary(tree.left, boundary)
        
    if tree.right != None:
        boundary = find_boundary(tree.right, boundary)
    
    return boundary
    
def predict_data(target, boundary_list):
    for boundary in boundary_list:
        if target[0] >= boundary[0] and target[0]<boundary[1] and target[1] >= boundary[2] and target[1] < boundary[3]:
            return boundary[4]
        
    
def cal_test_error(data_set, tree):
    boundary = []
    error_num = 0
    boundary = find_boundary(tree, boundary)
    n = len(boundary)
    for data in data_set:
        d_predict = predict_data(data, boundary)
        if d_predict != data[2]:
            error_num = error_num + 1
    
    error = error_num/len(data_set)
                    
    return n, error
    
def Q6_plot(data):
    fig = plt.figure()
    y_data = data[:,2]
    positive_data = data[y_data == 1]
    negative_data = data[y_data == 0]
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(positive_data[:,0], positive_data[:,1], c = 'blue', marker = 'o', label = "1")
    plt.scatter(negative_data[:,0], negative_data[:,1], c = 'red', marker = 'x', label = "0")
    plt.legend()

def Q7_dataset(data):
    np.random.seed(2)
    np.random.shuffle(data)
    d32 = data[0:31]
    d128 = data[0:127]
    d512 = data[0:511]
    d2048 = data[0:2047]
    d8192 = data[0:8191]
    dtest = data[8192:9999]
    return dtest, d32, d128, d512, d2048, d8192
    
    
##P2
#Q3
print("------Q3------")
tree = Tree()
HW2Q3 = True
data = read_data("Druns.txt")
tree = make_sub_tree(data, tree, 1)
HW2Q3 = False
print("------Q3------")

#Q4
print("------Q4------")
tree = Tree()
data = read_data("D3leaves.txt")
tree = make_sub_tree(data, tree, 1)
print_tree(tree)
print("------Q4------")

#Q5(1)
print("------Q5(1)------")
tree = Tree()
data = read_data("D1.txt")
tree = make_sub_tree(data, tree, 1)
print_tree(tree)
print("------Q5(1)------")


#Q5(2)
print("------Q5(2)------")
tree = Tree()
data = read_data("D2.txt")
tree = make_sub_tree(data, tree, 1)
print_tree(tree)
print("------Q5(2)------")

#Q6(1)
print("------Q6(1)------")
data_D1 = read_data("D1.txt")
Q6_plot(data_D1)

data_D2 = read_data("D2.txt")
Q6_plot(data_D2)
print("------Q6(1)------")

#Q6(2)
print("------Q6(2)------")
tree = Tree()
data_D1 = read_data("D1.txt")
tree = make_sub_tree_plot(data_D1, tree, 1,  0, 1, 0, 1)
plot_boundary(data_D1, tree)

tree = Tree()
data_D2 = read_data("D2.txt")
tree = make_sub_tree_plot(data_D2, tree, 1,  0, 1, 0, 1)
plot_boundary(data_D2, tree)
print("------Q6(2)------")

#Q7
print("------Q7------")
data = read_data("Dbig.txt")
dtest, d32, d128, d512, d2048, d8192 = Q7_dataset(data)

tree32 = make_sub_tree_plot(d32, Tree(), 1, -1.5, 1.5, -1.5, 1.5)
tree128 = make_sub_tree_plot(d128, Tree(), 1, -1.5, 1.5, -1.5, 1.5)
tree512 = make_sub_tree_plot(d512, Tree(), 1, -1.5, 1.5, -1.5, 1.5)
tree2048 = make_sub_tree_plot(d2048, Tree(), 1, -1.5, 1.5, -1.5, 1.5)
tree8192 = make_sub_tree_plot(d8192, Tree(), 1, -1.5, 1.5, -1.5, 1.5)

plot_boundary(d32, tree32)
plot_boundary(d128, tree128)
plot_boundary(d512, tree512)
plot_boundary(d2048, tree2048)
plot_boundary(d8192, tree8192)


n32, error32 = cal_test_error(dtest, tree32)
n128, error128 = cal_test_error(dtest, tree128)
n512, error512 = cal_test_error(dtest, tree512)
n2048, error2048 = cal_test_error(dtest, tree2048)
n8192, error8192 = cal_test_error(dtest, tree8192)

print("n",(n32,n128,n512,n2048, n8192),"error",(error32, error128, error512, error2048, error8192))

plt.figure()
plt.ylabel("err")
plt.xlabel("n")
plt.plot((32,128,512,2048, 8192),(error32, error128, error512, error2048, error8192))

print("------Q7------")


