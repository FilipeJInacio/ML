import numpy as np
import matplotlib as plot
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV

###########################################################

#   - Experimentar Ridge e Lasso -- Kinda done
#   - Experimentar Cross Validation e Leave One Out
#   - Ter cuidado com over-adjustement
#   - Diferenciar trabalho da task2 com a task1

###########################################################

# n amostras, 10 dimensões
x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")
design_matrix = np.hstack((np.ones((len(x), 1)), x))
beta = np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(y)
sse = np.sum((np.linalg.norm(y - design_matrix.dot(beta))) ** 2)

expected_y = design_matrix.dot(beta)
print("Predictor Results: \n", expected_y)
print("\n")
print("Actual Results: \n", y)

print("\n")
print("Associated Error is: %.4f" % sse)

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
print(" R squared Lasso is", round(lasso.score(x, y), 5))
print(" R squared Ridge is", round(ridge.score(x, y), 5))
print(" R squared LassoCV is", round(lasso_cv.score(x, y), 5))
print(" R squared RidgeCV is", round(ridge_cv.score(x, y), 5))

design_matrix_test = np.hstack((np.ones((len(x_test), 1)), x_test))
expected_y_test = design_matrix_test.dot(beta)
# print("Predictor Results: \n", expected_y_test)
