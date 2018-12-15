from DataReader import DataReader
import matplotlib.pyplot as plt
import numpy as np

data_reader = DataReader('./data/cifar-100-python')
tr_data, tr_class_lables, tr_subclass_labels = data_reader.get_train_data()
data_reader.plot_random(tr_data,25)

