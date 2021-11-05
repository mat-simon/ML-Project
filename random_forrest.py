import numpy as np
import scipy.io


def main():
    dictionary_data = scipy.io.loadmat('MNISTmini.mat')
    print("dictionary keys:", dictionary_data.keys())

    train_fea1 = np.array(dictionary_data['train_fea1'])
    print("shape of train_fea1 np array:", train_fea1.shape)

    train_gnd1 = np.array(dictionary_data['train_gnd1'])
    print("shape of train_gnd1:", train_gnd1.shape)

    test_fea1 = np.array(dictionary_data['test_fea1'])
    print("shape of test_fea1:", test_fea1.shape)

    test_gnd1 = np.array(dictionary_data['test_gnd1'])
    print("shape of test_gnd1:", test_gnd1.shape)


if __name__ == '__main__':
    main()
