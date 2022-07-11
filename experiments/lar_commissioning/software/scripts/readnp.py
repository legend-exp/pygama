import numpy as np
import matplotlib.pyplot as plt
import math

home_dir = "/home/pkrause/software/pygama/experiments/lar_commissioning/"
integ = [[]]*666
for i in range(666):
    integ[i] = np.load('th4kbqspe{}.npy'.format(i))

integ =np.concatenate(integ)
binwidth=100
plt.hist(integ,bins=range(0, math.ceil(max(integ)) + binwidth, binwidth))
plt.savefig('th4kbqspe.png')  
plt.show()