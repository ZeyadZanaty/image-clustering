import pickle
import numpy as np
from os.path import join
from os import listdir
import matplotlib.pyplot as plt

class DataReader:

    def __init__(self,root_dir,type='cifar-100'):
        self.root_dir = root_dir
        self.type = type
    
    def get_dict_from_pickle(self):
            self.train_dict = unpickle(join(self.root_dir,'train'))
            self.test_dict = unpickle(join(self.root_dir,'test'))
    
    def get_train_data(self):
        if self.type == 'cifar-100':
            self.get_dict_from_pickle()
            data = np.array(self.train_dict[b'data'])
            lbls_sub = np.array(self.train_dict[b'fine_labels'])
            lbls_class = np.array(self.train_dict[b'coarse_labels'])
            return data,lbls_class,lbls_sub
        elif self.type == 'cifar-10':
            data = np.empty(shape=(0,3072))
            labels = np.empty(shape=(0,))
            for file_ in listdir(self.root_dir):
                if file_.split('_')[0] == 'data':
                    dict = unpickle(join(self.root_dir,file_))
                    data = np.vstack((data,dict[b'data']))
                    labels  = np.hstack((labels,dict[b'labels']))
            return np.array(data),np.array(labels),None

    def get_test_data(self):
        if self.type == 'cifar-100':
            self.get_dict_from_pickle()
            data = np.array(self.test_dict[b'data'])
            lbls_sub = np.array(self.test_dict[b'fine_labels'])
            lbls_class = np.array(self.test_dict[b'coarse_labels'])
            return data,lbls_class,lbls_sub
        elif self.type == 'cifar-10':
            data = np.empty(shape=(0,3072))
            labels = []
            for file_ in listdir(self.root_dir):
                if file_.split('_')[0] == 'test':
                    dict = unpickle(join(self.root_dir,file_))
                    data = np.vstack((data,dict[b'data']))
                    print(data[data.shape[0]-1])
                    labels.append(dict[b'labels'])
            return np.array(data),np.array(labels),None
    
    def reshape_to_plot(self,data):
        return data.reshape(data.shape[0],3,32,32).transpose(0,2,3,1).astype("uint8")

    def plot_random(self,data,n):
        data = self.reshape_to_plot(data)
        x1 = min(n//2,5)
        y1 = n//x1
        x = min(x1,y1)
        y = max(x1,y1)
        fig, ax = plt.subplots(x,y,figsize=(3,3))
        for j in range(x):
            for k in range(y):
                i = np.random.choice(range(len(data)))
                ax[j][k].set_axis_off()
                ax[j][k].imshow(data[i:i+1][0])
        plt.show()

    def plot_imgs(self,in_data,n):
        data = np.empty(shape=(0,3072))
        for d in in_data:
            data = np.vstack((data,np.array(d)))
        data = self.reshape_to_plot(data)
        x1 = min(n//2,5)
        y1 = n//x1
        x = min(x1,y1)
        y = max(x1,y1)
        fig, ax = plt.subplots(x,y,figsize=(3,3))
        i=0
        for j in range(x):
            for k in range(y):
                ax[j][k].set_axis_off()
                ax[j][k].imshow(data[i:i+1][0])
                i+=1
        plt.show()
    
    def plot_img(self,data):
        assert data.shape == (3072,)
        data = data.reshape(1,3072)
        data = data.reshape(data.shape[0],3,32,32).transpose(0,2,3,1).astype("uint8")
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(data[0])
        plt.show()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict