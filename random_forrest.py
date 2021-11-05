import numpy as np
import scipy.io

dictionary_data = scipy.io.loadmat('MNISTmini.mat')

print("dictionary keys:", dictionary_data.keys())
train_fea1 = np.array(dictionary_data['train_fea1'])
print("size of train_fea1 np array: ", train_fea1.size)
print(train_fea1)

train_gnd1 = np.array(dictionary_data['train_gnd1'])
print("=========================================")
print("size of train_gnd1:", train_gnd1.size)
print(train_gnd1)

test_fea1 = np.array(dictionary_data['test_fea1'])
print("=========================================")
print("size of test_fea1", test_fea1.size)
print(test_fea1)

test_gnd1 = np.array(dictionary_data['test_gnd1'])
print("=========================================")
print("size of test_gnd1", test_gnd1.size)
print(test_gnd1)
