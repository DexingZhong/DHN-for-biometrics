#-------------------------------------
# Project: DHN for biometrics
# Date: 2019.11.02
# Author: Huikai Shao
# All Rights Reserved
#-------------------------------------
import numpy as np
import time


data_txt = open('feature.txt')
data_paths = data_txt.readlines()


total_picture=600

features = []
temp = np.array([])
for data_path in data_paths:
    temp = np.loadtxt(data_path[:-1])
    temp_list = list(temp)
    features.append(temp)
features = np.array(features)

def main1():
    features1 = []
    temp = np.array([])
    for data_path in data_paths:
        temp = np.loadtxt(data_path[:-1])
        temp_list = list(temp)
        features1.append(temp)
    features1 = np.array(features1)
    sums = []
    for i,feature in enumerate(features1):
        start_time = time.time()
        for feature1 in features:
            sum_array = np.sum(np.fabs(feature - feature1))/2
            sums.append(sum_array)
        duration = time.time() - start_time
        print(i,duration)
    results = np.reshape(sums,(total_picture,total_picture))
    np.savetxt("./data1.txt",results,fmt="%d")

if __name__ == '__main__':

    main1()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
