import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV

# n amostras, 10 dimensões
alphas = np.linspace(
    0.00000001, 0.001, 5000
)  # (valor inicial, valor final, n.º de prontos)
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

""" x_test = np.load("X_test_regression1.npy")

design_matrix_test = np.hstack((np.ones((len(x_test), 1)), x_test))
expected_y_test = design_matrix_test.dot(beta)
# print("Predictor Results: \n", expected_y_test) """


alphas_ridge = np.arange(0.01, 5, 0.001)
param_grid = {"alpha": alphas_ridge}
grid_ridge = Ridge()
grid_search = GridSearchCV(grid_ridge, param_grid, cv=5, verbose=10)
grid_search.fit(x, y)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
ridge_model = Ridge(alpha=3.348999999999997)
ridge_model.fit(x, y)
print(ridge_model.score(x, y))
results = cross_validate(ridge_model, x, y, cv=5)
cv_score = cross_val_score(ridge_model, x, y, cv=5)
print(cv_score.mean())
print(np.mean(results["test_score"]))
