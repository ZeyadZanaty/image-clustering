import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

class DataReader:

    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.get_dict_from_pickle()
    
    def get_dict_from_pickle(self):
        self.train_dict = unpickle(join(self.root_dir,'train'))
        self.test_dict = unpickle(join(self.root_dir,'test'))
    
    def get_train_data(self):
        data =  np.array(self.train_dict[b'data'])
        # data_for_plot =  self.reshape_to_plot(data)
        lbls_class = np.array(self.train_dict[b'fine_labels'])
        lbls_sub = np.array(self.train_dict[b'coarse_labels'])
        return data,lbls_class,lbls_sub

    def get_test_data(self):
        data =  self.test_dict[b'data']
        # data_for_plot =  self.reshape_to_plot(data)
        lbls_class = np.array(self.test_dict[b'fine_labels'])
        lbls_sub = np.array(self.test_dict[b'coarse_labels'])
        return data,lbls_class,lbls_sub
    
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

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict