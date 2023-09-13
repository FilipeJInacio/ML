import numpy as np

small_x = np.array([1, 2, 3])
big_x = np.hstack((np.ones((n, 1)), np.array([[x,1]]).T))
x = np.array([24, 30, 36])
y = np.array([13, 14, 16])

cost = np.sum(np.square(np.norm()))

