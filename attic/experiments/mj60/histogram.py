import numpy as np
import matplotlib.pyplot as plt

# this code is not complete. still want to make more general using pygama functions.

file1 = np.load("./psd_214_bl.npz")
file2 = np.load("./psd_249_bl.npz")

x1, y1 = file1["arr_0"], file1["arr_1"]
x2, y2 = file2["arr_0"], file2["arr_1"]

plt.semilogy(x1, y1, '-r', label='Run 214')
plt.semilogy(x2, y2, '-b', label='Run 249')

plt.title('Baseline PSD')
plt.xlabel('Frequency (Hz)', ha='right', x=0.9)
plt.ylabel('PSD (ADC^2 / Hz)', ha='right', y=1)
plt.legend(loc=1)
plt.tight_layout()
plt.show()
exit()
