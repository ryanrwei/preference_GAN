from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from itertools import permutations
from sklearn.datasets import make_blobs
import random
from utils import resize_to_ori_calMRE

class Kmeans:
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.num_iter = 0
        
    def calc_distance(self,x1,x2):
        diff = x1 - x2
        distances = np.sqrt(np.square(diff).sum(axis=1))
        return distances        

    def calc_distance_emd_hist(self, x1, x2, m):
        np.set_printoptions(precision=2)
        # x1, x2 are two election histograms (normalized)
        # m is number of alternatives

        fact_m = len(x1)

        # find mapping
        alts = tuple(range(m))
        perms = list(permutations(alts))

        emd = np.inf

        for i, p1 in enumerate(perms):

            new_x2 = []
            for p2 in perms:

                new_p2 = []
                for j in p2:
                    new_p2.append(p1[j])
                new_p2 = tuple(new_p2)
                idx = perms.index(new_p2)

                new_x2.append(x2[idx])
            new_x2 = np.array(new_x2)

            emd = np.min((emd, np.linalg.norm(x1 - new_x2)))

        return emd

    
    def fit(self, x, num_alternative, max_iter):
        self.x = x
        num_samples = self.x.shape[0]
        num_features = self.x.shape[1]
        self.num_alternative = num_alternative
        
        # Kmeans++ select center 
        first = np.random.choice(num_samples)
        # init_center list
        index_select = [first]
        # cal the rest k-1 centers
        for i in range(1, self.num_cluster):
            all_distances = np.empty((num_samples,0))
            for j in index_select:
                # calculate the distance between each point and selected center
                distances = []
                for k in range(len(self.x)):
                    distances.append(self.calc_distance_emd_hist(self.x[k], x[j], self.num_alternative))
                distances = np.array(distances).reshape([-1, 1])                 
                # store the distance between each point and selected center in an array, each col store one selected center 
                all_distances = np.c_[all_distances, distances]
            # Find the minimum distance from each point to the selected center of mass
            min_distances = all_distances.min(axis=1).reshape(-1,1)
            # select the most farthest point as new center
            index = np.argmax(min_distances)
            index_select.append(index)
        self.original_center = x[index_select]
#         print('init finish')
        
        while True and self.num_iter <= max_iter :
            # initialize a dict, taks cluster as key and assign it an empty array
            dict_y = {}
            for j in range(self.num_cluster):
                dict_y[j] = np.empty((0,num_features))
            for i in range(num_samples):
                distances = []
                for j in range(len(self.original_center)):
                    distances.append(self.calc_distance_emd_hist(x[i], self.original_center[j], self.num_alternative))
                distances = np.array(distances).reshape([-1])  

                # assign x[i] into the most closed center, store it in a dict
                label = np.argsort(distances)[0]
                dict_y[label] = np.r_[dict_y[label],x[i].reshape(1,-1)]
            centers = np.empty((0,num_features))
            # re-calculalte the center of each cluster 
            for i in range(self.num_cluster):
                center = np.mean(dict_y[i],axis=0).reshape(1,-1)
                centers = np.r_[centers,center]
            # if centers[i] == centers[i+1]: stop the training
            result = np.all(centers == self.original_center)
            if result == True:
                break
            else:
                # update centers
                self.original_center = centers
                
            self.num_iter += 1
#             print('current num_iter: ', self.num_iter)

        print('total used num_iter: ', self.num_iter)

    def predict_and_cluster(self, x):
        y_preds = []
        num_samples = x.shape[0]
        self.num_features = x.shape[1]
        
        cluster_x = {}
        for i in range(self.num_cluster):
            cluster_x[i] = []
        
        for i in range(num_samples):
            distances = []
            for j in range(len(self.original_center)):
                distances.append(self.calc_distance_emd_hist(x[i], self.original_center[j], self.num_alternative))
            distances = np.array(distances).reshape([-1]) 
            
            y_pred = np.argsort(distances)[0]
            y_preds.append(y_pred)
            
            cluster_x[y_pred].append(x[i])
            
        return y_preds, cluster_x
    
    def Kmeans_sampling_fn(self, x, num_sampling):
        num_sampling = int(num_sampling//self.num_cluster)
        
        sampled_x = []
        for i in range(self.num_cluster):
            random_seed = np.random.randint(len(x[i]), size = num_sampling)
            sampled_x.append(np.array(x[i])[random_seed])
        sampled_x_ = np.array(sampled_x).reshape([-1, self.num_features])
        np.random.seed(0)
        np.random.shuffle(sampled_x_)
        
        return sampled_x_