from DataReader import DataReader
from KMeans import KMeans
import matplotlib.pyplot as plt
import numpy as np

data_reader = DataReader('./data/cifar-100-python')
tr_data, tr_subclass_labels, tr_class_labels = data_reader.get_train_data()
# data_reader.plot_random(tr_data,25)
tr_data_slice = tr_data[:5000]
tr_labels_slice = tr_class_labels[:5000]
kmeans = KMeans(n_clusters=10,max_iter=20)
kmeans.fit(tr_data_slice,tr_labels_slice)
plt.plot(range(kmeans.iterations),kmeans.loss_per_iteration)
plt.show()