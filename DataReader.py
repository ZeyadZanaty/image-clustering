import pickle
import numpy as np
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct as st
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
            data = []
            labels = []
            print("Reading")
            for file_ in tqdm(listdir(self.root_dir)):
                if file_.split('_')[0] == 'data':
                    dict = unpickle(join(self.root_dir,file_))
                    data.extend(dict[b'data'])
                    labels.extend(dict[b'labels'])
            return np.array(data),np.array(labels),None
        elif self.type =='mnist':
            return self.read_mnist()

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
        if self.type == 'mnist':
            return data.reshape(data.shape[0],28,28).astype("uint8")
        return data.reshape(data.shape[0],3,32,32).transpose(0,2,3,1).astype("uint8")

    def plot_imgs(self,in_data,n,random=False):
        data = np.array([d for d in in_data])
        data = self.reshape_to_plot(data)
        x1 = min(n//2,5)
        if x1 == 0:
            x1 = 1
        y1 = (n//x1)
        x = min(x1,y1)
        y = max(x1,y1)
        fig, ax = plt.subplots(x,y,figsize=(5,5))
        i=0
        for j in range(x):
            for k in range(y):
                if random:
                    i = np.random.choice(range(len(data)))
                ax[j][k].set_axis_off()
                ax[j][k].imshow(data[i:i+1][0])
                i+=1
        plt.show()
    
    def plot_img(self,data):
        if self.type !='mnist':
            assert data.shape == (3072,)
            data = data.reshape(1,3072)
            data = data.reshape(data.shape[0],3,32,32).transpose(0,2,3,1).astype("uint8")
        elif self.type == 'mnist':
            assert data.shape == (28*28,)
            data = data.reshape(1,28,28).astype('uint8')
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(data[0])
        plt.show()
    
    def read_mnist(self):  
        filename = {'images' : 'train-images-idx3-ubyte' ,'labels' : 'train-labels-idx1-ubyte'}
        labels_array = np.array([])
        data_types = {
                0x08: ('ubyte', 'B', 1),
                0x09: ('byte', 'b', 1),
                0x0B: ('>i2', 'h', 2),
                0x0C: ('>i4', 'i', 4),
                0x0D: ('>f4', 'f', 4),
                0x0E: ('>f8', 'd', 8)}
        for name in filename.keys():
            if name == 'images':
                imagesfile = open(join(self.root_dir,filename[name]),'rb')
            if name == 'labels':
                labelsfile = open(join(self.root_dir,filename[name]),'rb')
        imagesfile.seek(0)
        magic = st.unpack('>4B',imagesfile.read(4))
        if(magic[0] and magic[1])or(magic[2] not in data_types):
            raise ValueError("File Format not correct")
        nDim = magic[3]
        imagesfile.seek(4)
        nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images/labels
        nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
        nC = st.unpack('>I',imagesfile.read(4))[0] #num of columns
        nBytes = nImg*nR*nC
        labelsfile.seek(8) #Since no. of items = no. of images and is already read
        images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytes,imagesfile.read(nBytes))).reshape((nImg,nR,nC))
        labels_array = np.asarray(st.unpack('>'+'B'*nImg,labelsfile.read(nImg))).reshape((nImg,1))
        labels_array = [l[0] for l in labels_array]
        return images_array.reshape(60000,28*28),labels_array,None


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict