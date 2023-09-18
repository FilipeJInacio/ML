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
x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")
design_matrix = np.hstack((np.ones((len(x), 1)), x))
beta = np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(y)
SSE = np.sum((np.linalg.norm(y - design_matrix.dot(beta))) ** 2)

expected_y = design_matrix.dot(beta)
print("Predictor Results: \n", expected_y)
print("\n")
print("Actual Results: \n", y)

print("\n")
print("Associated Error is: %.4f" % SSE)

lasso = Lasso(alpha=0.005)
lasso_cv = LassoCV(n_alphas=3)
ridge = Ridge(alpha=0.005)
ridge_cv = RidgeCV()
lasso.fit(x, y)
lasso_cv.fit(x, y)
ridge.fit(x, y)
ridge_cv.fit(x, y)

print(lasso.predict(x))
print(ridge.predict(x))
print(lasso_cv.predict(x))
print(ridge_cv.predict(x))

x_test = np.load("X_test_regression1.npy")
# print(" R squared ", round(lasso.score(x, y) * 100, 2))
# print(" R squared ", round(ridge.score(x, y) * 100, 2))

design_matrix_test = np.hstack((np.ones((len(x_test), 1)), x_test))
expected_y_test = design_matrix_test.dot(beta)
# print("Predictor Results: \n", expected_y_test)
