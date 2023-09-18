import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
import matplotlib as mpl
from matplotlib import rc

# Add the path to the 'latex' executable to the PATH
latex_path = "/usr/local/texlive/2023/bin/x86_64-linux"
os.environ["PATH"] = f"{latex_path}:{os.environ['PATH']}"


# Configure Matplotlib to use LaTeX for rendering
mpl.rcParams["text.usetex"] = True
mpl.rcParams[
    "text.latex.preamble"
] = r"\usepackage{amsmath}"  # You can add more packages if needed

# Enable LaTeX rendering
rc("text", usetex=True)

###########################################################

#   - Experimentar Ridge e Lasso -- Kinda done
#   - Experimentar Cross Validation e Leave One Out
#   - Ter cuidado com over-adjustement
#   - Diferenciar trabalho da task2 com a task1. Radial Basis Functions for task2 (?)
#   - Calcular R squared da Regressão Linear normal - conseguir a média dos y's para a variância e depois fazer o somatório pela expressão
#   - Para calcular os parâmetros alpha de cada modelo scikitlearn, existe uma função para a sua determinação mas fazendo um gráfico do erro em função
#   do alpha é a maneira recomendada pelo professor. No entanto no final usar a função para validar o valor obtido

###########################################################

# n amostras, 10 dimensões
# Set up alpha values
alphas = np.linspace(0.01, 1.0, 500)
r_values_lasso = []
r_values_ridge = []
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

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)

    lasso.fit(x, y)
    ridge.fit(x, y)

    r_values_lasso.append(lasso.score(x, y))
    r_values_ridge.append(ridge.score(x, y))

# print(lasso.predict(x))
# print(ridge.predict(x))
# print(lasso_cv.predict(x))
# print(ridge_cv.predict(x))

x_test = np.load("X_test_regression1.npy")
print(" R squared Lasso is", round(lasso.score(x, y), 5))
print(" R squared Ridge is", round(ridge.score(x, y), 5))
# print(" R squared LassoCV is", round(lasso_cv.score(x, y), 5))
# print(" R squared RidgeCV is", round(ridge_cv.score(x, y), 5))

design_matrix_test = np.hstack((np.ones((len(x_test), 1)), x_test))
expected_y_test = design_matrix_test.dot(beta)
# print("Predictor Results: \n", expected_y_test)

# Plot R-squared values for Lasso and Ridge
plt.plot(alphas, r_values_lasso, label=r"Lasso")
plt.plot(alphas, r_values_ridge, label=r"Ridge")
plt.xlabel(r"\textbf{Alpha [$\alpha$]}")
plt.ylabel(r"\textbf{Coefficient of Determination [$R^2$]}")
plt.title(r"R-squared vs Alpha for Lasso and Ridge")
plt.legend()
plt.show()
