import numpy as np
import pandas as pd


# one hot code
def one_hot ( labels , Label_class ): 
    one_hot_label = np.array([[ int (i == int (labels[j])) for i in range (Label_class)] for j in range ( len (labels))])      
    return one_hot_label

def resize_to_ori_calMRE(x, img_size, img_width, img_height, ori_size):
    x = x.reshape([-1, img_width*img_height])
    cell_size = int(img_size//ori_size)

    ori_dat = []
    for j in range(x.shape[0]):
        dat = x[j]
        tmp = []
        for i in range(dat.shape[0] // cell_size):
            k = dat[(i)*cell_size:(i+1)*cell_size]
            tmp.append(np.mean(k))
        ori_dat.append(tmp)
    ori_dat = np.array(ori_dat)    
    return ori_dat

def resize_to_ori_calMAE(x, img_size, img_width, img_height, ori_size):
    x = x.reshape([-1, img_width*img_height])
    cell_size = int(img_size//ori_size)

    ori_dat = []
    for j in range(x.shape[0]):
        dat = x[j]
        tmp = []
        for i in range(dat.shape[0] // cell_size):
            k = dat[(i)*cell_size:(i+1)*cell_size]
            tmp.append(np.mean(k))
        ori_dat.append(tmp)
    ori_dat = np.array(ori_dat)    
    return ori_dat

def resize_to_ori(x, img_size, img_width, img_height, ori_size, batch_size, iteration_generator):
    x = x.reshape([iteration_generator*batch_size, img_width*img_height])
    cell_size = int(img_size//ori_size)

    ori_dat = []
    for j in range(x.shape[0]):
        dat = x[j]
        tmp = []
        for i in range(dat.shape[0] // cell_size):
            k = dat[(i)*cell_size:(i+1)*cell_size]
            tmp.append(np.mean(k))
        ori_dat.append(tmp)
    ori_dat = np.array(ori_dat)    
    return ori_dat