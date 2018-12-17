from DataReader import DataReader
from KMeans import KMeans
import matplotlib.pyplot as plt
import numpy as np

data_reader = DataReader('./data/cifar-10-python','cifar-10')
tr_data, tr_class_labels, tr_subclass_labels = data_reader.get_train_data()
# data_reader.plot_random(tr_data,25)
tr_data_slice = tr_data[:50000]
tr_labels_slice = tr_class_labels[:50000]
kmeans = KMeans(n_clusters=10,max_iter=200)
kmeans.fit(tr_data_slice,tr_labels_slice)
data_reader.plot_imgs(kmeans.centroids,len(kmeans.centroids))
plt.plot(range(kmeans.iterations),kmeans.loss_per_iteration)
plt.show()

for key,data in list(kmeans.clusters['data'].items()):
    data_reader.plot_imgs(data[:min(25,data.shape[0])],min(25,data.shape[0]))