import numpy as np
class KMeans:
    
    def __init__(self,n_clusters=5,max_iter=500):
        self.centroids = []
        self.n_clusters = n_clusters
        self.max_iter = max_iter