import numpy as np
class KMeans(object):
    """
    Implementation of KMeans clustering algorithm
    """
    def __init__(self, k, distance = "euclidean", init_strategy = "kmeans++", threshold = .0001, max_iter = 100):
        self.k = k
        self.distance = distance
        self.init_strategy = init_strategy
        self.threshold = threshold
        self.max_iter = max_iter
        self.centroids = None
        self.mse = None
    
    def _find_distance(self, point1, point2):
        """
        Find distances between points in n-dimensional space using the chosen distance metric.
        
        Parameters
        ----------
        point1 : NumPy Array
        A single observation which you'd like to find distances for
        
        point2 : NumPy Array
        Either a single observation, or an array of observations to compare to point1
        
        Returns
        -------
        distances : NumPy Array
        A scalar value indicating the distance between point1 and point2 if point2 is a single observation;
        otherwise, an array of scalars of the same length as point2 containing the pairwise distances from point1
        """
        if self.distance in ["euclidean", "l2"]:
            if len(point2.shape) > 1:
                distance = np.sqrt(np.sum((point1 - point2) ** 2, axis = 1))
            else:
                distance = np.sqrt(np.sum((point1 - point2) ** 2))
        
        elif self.distance in ["manhattan", "l1"]:
            if len(point2.shape) > 1:
                distance = np.abs(point1 - point2).sum(axis = 1)
            else:
                distance = np.abs(point1 - point2).sum()
        
        elif self.distance in ["cosine", "cosine_similarity"]:
            if len(point2.shape) > 1:
                #Had to use np.maximum here because this would sometimes return an extremely small negative value instead of zero
                distance = np.maximum(0,1 - (np.sum(point1 * point2, axis = 1) / (np.linalg.norm(point1) * np.linalg.norm(point2, axis = 1))))
            else:
                distance = np.maximum(0,1 - (np.sum(point1 * point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))))
        
        else:
            raise ValueError("Unknown distance measure selected")
            
        return distance
    
    def _choose_initial(self, X):
        if self.init_strategy == "assign_randomly":
            
            assignments = np.random.randint(0, self.k, size = len(X))
            centroids = []
            for i in range(self.k):
                centroids.append(np.mean(X[assignments == i,:], axis = 0))
                
            centroids = np.array(centroids)
        
        elif self.init_strategy == "random_points":
            rand_indexes = np.random.choice(range(len(X)), size = self.k, replace = False)
            centroids = X[rand_indexes,:]
        
        elif self.init_strategy == "kmeans++":
            rand_index = np.random.choice(range(len(X)))
            centroids = X[rand_index,:].reshape(1,-1)
            
            for i in range(self.k - 1):
                distances = np.apply_along_axis(arr = centroids, axis = 1, func1d = self._find_distance, point2 = X)
                weights = distances.min(axis = 0)
                index = np.random.choice(range(len(X)), p = weights / np.sum(weights))
                centroids = np.vstack([centroids, X[index,:]])            
        
        else:
            raise ValueError("Unknown initialization strategy chosen")
        
        return centroids
    
    def _find_intra_cluster_variance(self, X, centroids, assignments):
        result = 0
        for i in range(self.k):
            result += np.sum((X[i == assignments,:] - centroids[i,:]) ** 2)
        
        return result
    
    def fit(self, X):
        """
        Fits k clusters to the supplied data using the distance and initialization schemes supplied in the constructor
        
        Parameters
        ----------
        X : NumPy Array
        Data to which you would like clusters to be fit
        
        Returns
        -------
        centroids : NumPy Array
        The values of the k converged centroids
        """
        centroid_history = []
        assignment_history = []
        centroids = self._choose_initial(X)
        centroid_history.append(centroids)
        done = False
        iterations = 0
        while not done and iterations < self.max_iter:
            #Calculate distances from each centroid
            distances = np.apply_along_axis(arr = centroids, axis = 1, func1d = self._find_distance, point2 = X)
            
            #Assign each point to the centroid which has the lowest distance
            assignments = distances.argmin(axis = 0)
            new_centroids = []
            for i in range(self.k):
                #There are some situations where a cluster is not closest to any of the points
                #For example, in the assign_randomly scheme, one cluster could be closely surrounded by all the other clusters
                #This is more likely than you'd think, since all the clusters tend to be around the mean of the dataset
                #Random_points also has this issue -- even though the sampling is done without replacement, there is still the
                #possibility of repeated data. In order to resolve this issue, I just assigned such a centroid to the mean of the dataset.
                if len(X[assignments == i,:]) == 0:
                    new_centroids.append(np.mean(X, axis = 0))
                else:
                    new_centroids.append(np.mean(X[assignments == i,:], axis = 0))
            new_centroids = np.array(new_centroids)
            
            #Keep track of some values for debugging / plotting
            centroid_history.append(new_centroids)
            assignment_history.append(assignments)
            
            #Calculate intra-cluster variance to see if we're ready to stop
            old_var = self._find_intra_cluster_variance(X, centroids, assignments)
            new_var = self._find_intra_cluster_variance(X, new_centroids, assignments)
    
            if np.abs(old_var - new_var) / old_var < self.threshold:
                done = True
            centroids = new_centroids
            
            iterations += 1
        
        self._assignment_history = np.array(assignment_history)
        self._centroid_history = np.array(centroid_history)
        self.centroids = centroids
        self.assignments = assignments
        self.mse = self._find_intra_cluster_variance(X, centroids, assignments)
        return self.centroids