import numpy as np
import matplotlib.pyplot as plt

# Create data
x = np.arange(0, 11)
y = x ** 3

plt.plot(x, y, 'r-')
plt.axis((0, 10, 0, 1000))

plt.show()
