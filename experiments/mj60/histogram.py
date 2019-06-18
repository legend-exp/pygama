import numpy as np
import matplotlib.pyplot as plt

file2 = np.load("./psd_stuff.npz")
file1 = np.load("./psd_stuff1.npz")

x1, y1 = file1["arr_0"], file1["arr_1"]
x2, y2 = file2["arr_0"], file2["arr_1"]

plt.semilogy(x1, y1, '-r', label='Run 220')
plt.semilogy(x2, y2, '-b', label='Older run')

plt.xlabel('Frequency (Hz)', ha='right', x=0.9)
plt.ylabel('PSD (ADC^2 / Hz)', ha='right', y=1)
plt.legend(loc=1)
plt.tight_layout()
plt.show()
exit()
