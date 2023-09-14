import numpy as np
import matplotlib as plot
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV

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
beta = np.linalg.inv(big_x.T.dot(big_x)).dot(big_x.T).dot(y)
SSE = np.sum((np.linalg.norm(y - big_x.dot(beta))) ** 2)

expected_y = big_x.dot(beta)
print("Predictor Results: \n", expected_y)
print("\n")
print("Actual Results: \n", y)

print("\n")
print("Associated Error is: %.4f" % SSE)

lasso = Lasso(alpha=0.005)
lasso_cv = LassoCV(n_alphas=3)
ridge = Ridge(alpha=0.005)
ridge_cv = RidgeCV()
lasso.fit(small_x, y)
lasso_cv.fit(small_x, y)
ridge.fit(small_x, y)
ridge_cv.fit(small_x, y)

print(lasso.predict(small_x))
print(ridge.predict(small_x))
print(lasso_cv.predict(small_x))
print(ridge_cv.predict(small_x))

small_x_test = np.load("X_test_regression1.npy")
# print(" R squared ", round(lasso.score(small_x, y) * 100, 2))
# print(" R squared ", round(ridge.score(small_x, y) * 100, 2))

big_x_test = np.hstack((np.ones((len(small_x_test), 1)), small_x_test))
expected_y_test = big_x_test.dot(beta)
# print("Predictor Results: \n", expected_y_test)
