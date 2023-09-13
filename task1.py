import numpy as np

# n amostras, 10 dimensiões
small_x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")
big_x = np.hstack((np.ones((len(small_x), 1)), small_x))

beta = np.linalg.inv(big_x.T.dot(big_x)).dot(big_x.T).dot(y)

SSE = np.sum((np.linalg.norm(y - big_x.dot(beta))) ** 2)
print("Beta 0: %.4f" % beta[0][0])
print("Beta 1: %.2f" % beta[1][0])
print("Associated Error is: %.4f" % SSE)

test_matrix = np.load("X_test_regression1.npy")

for i in test_matrix:
    for x_test in i:
        print("Predicted value for:", x_test)
        print(beta[0][0] + beta[1][0] * x_test)
        print("\n")
