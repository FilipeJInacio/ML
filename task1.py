import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    cross_validate,
    cross_val_score,
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# n amostras, 10 dimensões
alphas = np.linspace(0.00000001, 0.001, 5000)  # (valor inicial, valor final, n.º de prontos)
x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=21)

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


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

N = x_train.shape[0]
alphas_ridge = np.arange(0.01, 5, 0.1)
param_grid = {'alpha': alphas_ridge}

grid_ridge = Ridge()
grid_search = GridSearchCV(grid_ridge, param_grid, scoring='neg_mean_squared_error', cv=2, verbose=10)
grid_search.fit(x, y)

print(grid_search.best_estimator_)
print(grid_search.best_params_)

ridge_model = Ridge(alpha=3.999)
ridge_model.fit(x_train_scaled, y_train)

# print(ridge_model.score(x, y))

cv_results_sse = cross_val_score(ridge_model, x_train_scaled, y_train, cv=2, scoring='neg_mean_squared_error', error_score='raise')
cv_results_r2 = cross_val_score(ridge_model, x_train_scaled, y_train, cv=2, scoring='r2', error_score='raise')
cv_results = cross_validate(ridge_model, x_train_scaled, y_train, cv=2, scoring=['r2', 'neg_mean_squared_error'], return_train_score=True, error_score='raise')

print(f"R2 Validate:{(cv_results['test_neg_mean_squared_error'].mean())*15} \n R2 Score:{cv_results_r2.mean()} \n")

SSE = abs(cv_results_sse.mean()) * N
r2 = cv_results_r2.mean()
r2_true = cv_results

print(f"SSE:{SSE} \n R2:{r2} \n True R2:{r2_true} \n")
print(f"Predicted:{ridge_model.predict(x_test)} \n Real:{y_test}")
