import numpy as np
import matplotlib as plot
from sklearn.linear_model import Lasso, Ridge

###########################################################

#   - Experimentar Ridge e Lasso
#   - Experimentar Cross Validation e Leave One Out
#   - Ter cuidado com over-adjustement
#   - Diferenciar trabalho da task2 com a task1

###########################################################

# n amostras, 10 dimensiões
small_x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")
big_x = np.hstack((np.ones((len(small_x), 1)), small_x))

# print(y)

beta = np.linalg.inv(big_x.T.dot(big_x)).dot(big_x.T).dot(y)

beta_t = np.transpose(beta)

SSE = np.sum((np.linalg.norm(y - big_x.dot(beta))) ** 2)

big_x_t = np.transpose(big_x)

print(big_x.shape)

###################################

# beta_t = [beta_0  beta_1  beta_2  beta_3  beta_4  beta_5  beta_6  beta_7  beta_8  beta_9  beta_10]
# big_x

###################################

expected_y = beta.T.dot(big_x.T)
print(expected_y)

print("\n")

print("Beta 0: %.4f" % beta[0][0])
print("Beta 1: %.2f" % beta[1][0])
print("Associated Error is: %.4f" % SSE)

lasso = Lasso(alpha=0.005)
ridge = Ridge(alpha=0.005)
lasso.fit(small_x, y)
ridge.fit(small_x, y)

test_matrix = np.load("X_test_regression1.npy")
print(" R squared ", round(lasso.score(small_x, y) * 100, 2))
print(" R squared ", round(ridge.score(small_x, y) * 100, 2))

# for i in test_matrix:
#    for x_test in i:
#        print("Predicted value for:", x_test)
#        print(beta[0][0] + beta[1][0] * x_test)
#        print("\n")
