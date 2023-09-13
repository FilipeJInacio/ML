import numpy as np


x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")
X = np.hstack((np.ones((len(x), 1)), x))

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

SSE = np.sum((np.linalg.norm(y - X.dot(beta))) ** 2)

print(SSE)

"""
small_x = np.array([[24], [30], [36]])
y = np.array([[13], [14], [16]])
big_x = np.hstack((np.ones((3, 1)), small_x))

beta = np.array([[0], [0]])

beta = np.linalg.inv(big_x.T.dot(big_x)).dot(big_x.T).dot(y)

SSE = np.sum((np.linalg.norm(y - big_x.dot(beta))) ** 2)
print("Beta 0: %.4f" % beta[0][0])
print("Beta 1: %.2f" % beta[1][0])
print("Associated Error is: %.4f" % SSE)

print("Predicted value for 25: %.2f" % (beta[0][0] + beta[1][0] * 25))
print("Predicted value for 34: %.2f" % (beta[0][0] + beta[1][0] * 34))
"""