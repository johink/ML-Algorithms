import numpy as np

class DecisionTree(object):
    """
    DecisionTree allows the creation of classification trees containing both numeric and categorical feature variables.
    It also allows response variables with more than two classes.
    Besides the above, DecisionTree is designed to have a similar interface to scikit-learn.
    """
    class Node(object):
        #Nodes are the recursive structures that comprise the DecisionTree object
        def __init__(self, split_col, split_val, categorical, left, right, depth, classLeft, classRight):
            self.split_col = split_col
            self.split_val = split_val
            self.categorical = categorical
            self.left = left
            self.right = right
            self.depth = depth
            self.classLeft = classLeft
            self.classRight = classRight
        
        def toString(self):
            """
            Convenience function to recursively print tree.  Indentation indicates depth of node.
            """
            words = []
            if self.left:
                words.append(self.left.toString())

            words.append("     " * (self.depth - 1) + "-----" * min(1,self.depth) + "col: {} @ val: {}".format(self.split_col, self.split_val))

            if self.right:
                words.append(self.right.toString())
            
            return "\n".join(words)
        
        def predict(self, X):
            """
            Each node will predict only the rows for data which cannot be further split.
            Any data which can be further split will be passed along to the child nodes.
            """
            #Numerical features split left when less than split value, while categorical split left when equal to split value
            if self.categorical:
                mask = X[:,self.split_col] == self.split_val 
            else: 
                X[:,self.split_col] < self.split_val
            
            #If no left child, we must calculate the class probabilities for everything that is not going right
            if self.left is None:
                #Normalize the class counts into probabilities
                classDist = self.classLeft / self.classLeft.sum().astype(float)
                
                #Create an array of these probabilities of the same length as the number of observations
                #with the same number of columns as the number of classes
                #Reshape and transpose are used to get the output of np.repeat into the right format
                preds = np.repeat(classDist, len(X[mask])).reshape(len(classDist), len(X[mask])).T
                                 
                #Need a way to keep track of index.  DecisionTree predict method added an index as the rightmost column,
                #so we can use that to make sure the observations don't get jumbled
                labs = X[mask,-1].reshape(-1, 1)
                
                #Combine the probability predictions with the labels
                left = np.concatenate([preds, labs], axis = 1)                
            else:
                #If there is a left child, ask the left child what the predictions should be
                left = self.left.predict(X[mask])
                
            if self.right is None:
                classDist = self.classRight / self.classRight.sum().astype(float)
                preds = np.repeat(classDist, len(X[~mask])).reshape(len(classDist), len(X[~mask])).T
                labs = X[~mask,-1].reshape(-1, 1)
                right = np.concatenate([preds, labs], axis = 1)
            else:
                right = self.right.predict(X[~mask])
            
            return np.concatenate([left, right], axis = 0)
            
    
    def __init__(self, maxDepth = None, min_leaf_size = 1, min_gain = 0):
        """
        Parameters
        ----------
        maxDepth : int
        Maximum allowable depth of tree nodes
        
        min_leaf_size : int
        Minimum number of observations in a leaf node
        
        min_gain : float
        Entropy improvement threshold which must be met to conduct a split

        Returns
        -------
        dt : DecisionTree
        A newly-initialized DecisionTree object
        """
        self.root = None
        self.maxDepth = maxDepth
        self.min_leaf_size = min_leaf_size
        self.min_gain = min_gain
    
    @staticmethod
    def _entropy(x):
        """
        Calculate the entropy of a set of categorical data

        Parameters
        ----------
        x : NumPy Array
        Array containing class labels

        Returns
        -------
        entropy : float
        The entropy value of the array.  Valid values range between 0 and 1, inclusive.
        """
        classes = np.unique(x, return_counts = True)

        probs = classes[1] / classes[1].sum().astype(float)

        return -np.sum(probs * np.log2(probs))

    def _info_gain(self, x, x_left, x_right):
        """
        Calculate the entropy gain of a prospective split

        Parameters
        ----------
        x : NumPy Array
        Array containing class labels

        x_left, x_right : NumPy Arrays
        Partitions of x

        Returns
        -------
        gain : float
        The information gain of the proposed split
        """
        baseline = self._entropy(x)
        gain = baseline - ((len(x_left) / float(len(x))) * self._entropy(x_left) + 
                           ((len(x_right) / float(len(x))) * self._entropy(x_right)))
        return gain

    def _find_best_split(self, X, y):
        """
        Find the best split point for maximum entropy reduction

        Parameters
        ----------
        X : NumPy Array
        Two-dimensional array containing columns of data

        y : NumPy Array
        Vector of class labels

        Returns
        -------
        best_column : int
        The column with the optimal split

        best_value : depends on input array
        The value in best_column where the split should be conducted

        find_best_split returns None where there exists no split to reduce entropy
        """
        best_column = -1
        best_value = -1
        IG = 0

        if len(X.shape) == 1:
            X = X.reshape((-1,1))

        for col in range(X.shape[1]):
            #Mark the column as categorical if it's an instance of string or boolean
            categorical = any([isinstance(X[0,col], str), isinstance(X[0,col], bool)])
            for value in np.unique(X[:,col]):
                #Check for equality if categorical
                if categorical:
                    mask = X[:,col] == value
                #or < if not
                else:
                    mask = X[:,col] < value 
                
                #Calculate the information gain
                temp = self._info_gain(y, y[mask], y[~mask])
                
                #If this information gain is better than any of the others we've seen, remember it
                if temp > IG and len(y[mask]) >= self.min_leaf_size and len(y[~mask]) >= self.min_leaf_size:
                    IG = temp
                    best_column = col
                    best_value = value

        #If no column ever improved the information gain, or the gain is less than our threshold, return None
        if best_column == -1 or IG < self.min_gain:
            return None
        else:
            return best_column, best_value
        
    def _build_tree(self, X, y, depth, classes):
        #Stop growing the tree if we're too deep
        if self.maxDepth and depth > self.maxDepth:
            return None
        else:
            #Find the best split; if no best split, stop growing the tree here
            temp = self._find_best_split(X, y)
            if temp:
                best_col, best_val = temp
            else:
                return None
            
            categorical = any([isinstance(X[0,best_col], str), isinstance(X[0,best_col], bool)]) 
            if categorical:
                mask = X[:,best_col] == best_val
            else:
                mask = X[:,best_col] < best_val
            
            #Build out the child trees
            left = self._build_tree(X[mask,:], y[mask], depth + 1, classes)
            right = self._build_tree(X[~mask,:], y[~mask], depth + 1, classes)
            
            #Find the class probabilities for the child trees
            classLeft = np.vectorize(lambda x: (y[mask] == x).sum())(classes)            
            classRight = np.vectorize(lambda x: (y[~mask] == x).sum())(classes)
            
            #Create a new node in the tree.  Ultimately returns root
            return self.Node(best_col, best_val, categorical, left, right, depth, classLeft, classRight)

    def fit(self, X, y):
        """
        Generate the decision rules for classifying observations
        
        Parameters
        ----------
        X : NumPy Array
        Two-dimensional array containing columns of data
        
        y : NumPy Array
        Vector of class labels

        Returns
        -------
        self : DecisionTree
        A fitted DecisionTree object
        """
        classes = np.unique(y)
        self.root = self._build_tree(X, y, 0, classes)
        return self
        
    def predict_proba(self, X):
        """
        Return the probabilities for each target class
        
        Parameters
        ----------
        X : NumPy Array
        Two-dimensional array containing columns of data

        Returns
        -------
        predictions : NumPy Array
        Array containing the predicted class probabilities
        """
        temp = self.root
        if temp:
            #Here we concatenate index numbers to the data so that it may be returned in the same order it was received
            predictions = self.root.predict(np.concatenate([X, np.arange(len(X)).reshape(-1,1)], axis = 1))
            return np.array(sorted(predictions, key = lambda x: x[-1]))[:,:-1]
        else:
            return None
        
    def predict(self, X):
        """
        Return the class label that has the highest probability
        
        Parameters
        ----------
        X : NumPy Array
        Two-dimensional array containing columns of data

        Returns
        -------
        predictions : NumPy Array
        Array containing the predicted class labels
        """
        return np.argmax(self.predict_proba(X), axis = 1)
    
    def __str__(self):
        if self.root:
            return self.root.toString()
        
    def __repr__(self):
        return "DecisionTree(maxDepth = {}, min_leaf_size = {}, min_gain = {})".format(self.maxDepth, self.min_leaf_size, self.min_gain)