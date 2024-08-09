import numpy as np

# Euclidean metric
class Euclidean:
    def __init__(self):
        pass
    
    # Euclidean metric definition
    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    # cross distances of points in X
    def metric_aa(self, X):
        r = len(X) # number of rows (instances) in X
        distances = np.zeros(shape=(r,r), dtype=np.float32) # output matrix
        
        # triangular calculation of distances
        for i in range(r-1):
            for j in range(i+1, r):
                aux = self.distance(X[i], X[j]) # d(i,j) = d(j,i)
                distances[i,j] = aux
                distances[j,i] = aux
        
        return distances


# Collaborative filtering
class CollabFilt():
    def __init__(self, k=10):
        self.k = k # number of neighbors considered
        self.metric = Euclidean() # euclidean metric
        
    def fit(self, M_train):
        self.M_train = M_train

        # cross distances calculation
        self.Dmatrix = self.metric.metric_aa(self.M_train)

        #  prediction matrix construction
        self.M_pred = np.zeros(shape=self.M_train.shape, dtype=np.float32)
        for ui in range(self.M_train.shape[0]):
            knns = self.kNN(ui) # k Nearest Neighbors of ui
            auxM = self.M_train[knns] # info of kNN to ui
            
            # analisis per id
            for i, Mui in enumerate(self.M_train[ui,:]):
                # a prediction is calculated if no rating is given
                if Mui == 0.0:
                    # recolect information per item if exist
                    auxS = 0.0 # auxiliar sum
                    auxN = 0.0 # number of elements different of zero
                    for mi in auxM[:,i]: # item i information
                        if mi != 0.0:
                            auxS += mi
                            auxN += 1.0
                    
                    if auxN > 0:# calculate the mean over non-zero elements
                        mean_ui = auxS/auxN
                    else: # no information case
                        mean_ui = 0.0

                    # save the predicted mean
                    self.M_pred[ui,i] = mean_ui
        return self.M_pred
    
    # k-nearesth neighbors of ui
    def kNN (self, ui): 
        distances = self.Dmatrix[ui, 1:] # distances between ui and all the points on M_train-{ui}
        k_i = np.argsort(distances)[:int(self.k)] # indeces of the k neares neighbors of xi
        return k_i

    # predictions of ui    
    def predict(self, ui, topk=5):
        # top-k predictions of ui
        auxM = self.M_pred[ui]
        ind = np.argsort(auxM)
        return ind[-topk:] 