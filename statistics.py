#-------------------------------------
# Project: DHN for biometrics
# Date: 2019.11.02
# Author: Huikai Shao
# All Rights Reserved
#-------------------------------------
import numpy as np
import matplotlib.pyplot as plt

total_picture = 600
total_size = 10
train_size = 5

a = np.loadtxt("data1.txt",dtype = np.int32)
a = list(a)
true_list = []
false_list = []
for i in range(total_picture):
    if(i>=(int(i/total_size)*total_size+train_size) and i<(int(i/total_size)*total_size+total_size)):
        for j in range(total_picture):
            if(j>=(int(j/total_size)*total_size) and j<(int(j/total_size)*total_size+train_size)):
                if(int(i/total_size)==int(j/total_size)):
                    true_list.append(a[i][j])
                elif( i !=j ):
                    false_list.append(a[i][j])
true_numbers = []
false_numbers = []
true_number = 0
false_number = 0
print(len(true_list))
print(len(false_list))


for i in range(1,200):
    for k in true_list:
        if( k > i):
            true_number = true_number+1
    for k in false_list:
        if( k <= i):
            false_number = false_number +1
    true_numbers.append(true_number/1500) #500*6*1*6
    false_numbers.append(false_number/88500) #500*6*499*6
    true_number = 0
    false_number = 0

fig = plt.figure()
fig.clf()
ax = plt.subplot(111)
ax.plot([0,1],[0,1],'b--')
ax.plot(false_numbers, true_numbers, linewidth=3)
plt.xlabel('False Acceptance rate'); plt.ylabel('False Rejection rate') #600*9*599*3
ax.axis([0,1,0,1])
plt.show()


print(max(true_list))
print(min(false_list))
