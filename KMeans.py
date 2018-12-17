import numpy as np
import copy
from tqdm import tqdm
class KMeans:
    
    def __init__(self,n_clusters=10,max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = {'data':{i:None for i in range(n_clusters)}}
        self.clusters['labels']={i:None for i in range(n_clusters)}
        self.loss_ = []

    def init_centroids(self):
        np.random.seed(154)
        self.centroids = []
        for i in range(self.n_clusters):
            rand_index = np.random.choice(range(len(self.fit_data)))
            self.centroids.append(self.fit_data[rand_index])

    def fit(self,fit_data,fit_labels):
        self.fit_data = fit_data
        self.fit_labels = fit_labels
        self.predicted_labels = [None for _ in range(self.fit_data.shape[0])]
        self.init_centroids()
        self.iterations = 0
        old_centroids = [np.zeros(shape=(3072,)) for _ in range(self.n_clusters)]
        while not self.converged(self.iterations,old_centroids,self.centroids):
            print("\nIteration:",self.iterations)
            old_centroids = copy.deepcopy(self.centroids)
            for j,sample in tqdm(enumerate(self.fit_data)):
                min_dist = float('inf')
                for i,centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample-centroid)
                    if dist<min_dist:
                        min_dist = dist
                        self.predicted_labels[j] = i    
                if self.predicted_labels[j] is not None:
                    if self.clusters['data'][self.predicted_labels[j]] is None:
                        self.clusters['data'][self.predicted_labels[j]] = np.array(sample)
                    else:
                        self.clusters['data'][self.predicted_labels[j]] = np.vstack((self.clusters['data'][self.predicted_labels[j]],sample))
                    
                    if self.clusters['labels'][self.predicted_labels[j]] is None:
                        self.clusters['labels'][self.predicted_labels[j]] = []
                        self.clusters['labels'][self.predicted_labels[j]].append(self.fit_labels[j])
                    else:
                        self.clusters['labels'][self.predicted_labels[j]].append(self.fit_labels[j])
            self.update_centroids()
            self.calculate_loss()
            self.loss_.append(self.loss)
            print('Loss',self.loss)
            print('Difference',self.centroids_dist)
            self.iterations+=1

    def update_centroids(self):
        for i,cluster in enumerate(list(self.clusters['data'].values())):
            if type(cluster) is  None:
                self.centroids[i] = self.fit_data[np.random.choice(range(len(self.fit_data)))]
            else:
                self.centroids[i] = np.mean(cluster,axis=0)

    def converged(self,iterations,centroids,updated_centroids):
        if iterations > self.max_iter:
            return True
        self.centroids_dist = np.linalg.norm(np.array(centroids)-np.array(updated_centroids))
        if self.centroids_dist<0.5:
            return True
        return False

    def calculate_loss(self):
        self.loss = 0
        for key,value in list(self.clusters['data'].items()):
            for v in value:
                self.loss += np.linalg.norm(v-self.centroids[key])