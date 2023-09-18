import numpy as np

x = np.array([[24], [30], [36]])
y = np.array([[13], [14], [16]])
design_matrix = np.hstack((np.ones((3, 1)), x))

beta = np.array([[0], [0]])

beta = np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(y)

SSE = np.sum((np.linalg.norm(y - design_matrix.dot(beta))) ** 2)
print("Beta 0: %.4f" % beta[0][0])
print("Beta 1: %.2f" % beta[1][0])
print("Associated Error is: %.4f" % SSE)

print("Predicted value for 25: %.2f" % (beta[0][0] + beta[1][0] * 25))
print("Predicted value for 34: %.2f" % (beta[0][0] + beta[1][0] * 34))

# hello world
