import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import math



x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")

method = 8

if method == 1 or method == 0:
    X = np.hstack((np.ones((len(x), 1)), x))
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    prediction_loss = (np.linalg.norm(y - X.dot(beta))) ** 2
    sse = np.sum(prediction_loss)
    
    print("=============Least Squares=============")
    print("Error: %.4f" % sse)
    print(f"Beta: {beta}")
    
if method == 2 or method == 0:
    # Remove mean
    row_means = np.mean(x, axis=1)
    X = x - row_means[:, np.newaxis]
    Y = y - np.mean(y)
    
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    prediction_loss = (np.linalg.norm(y - X.dot(beta))) ** 2
    sse = np.sum(prediction_loss)

    print("=============Least Squares no mean=============")
    print("Error: %.4f" % sse)
    print(f"Beta: {beta}")

if method == 3 or method == 0:
    linear_regression = LinearRegression()
    linear_regression.fit(x, y)
    
    # Make predictions using the model
    y_pred = linear_regression.predict(x)

    sse = mean_squared_error(y, y_pred) * len(y)
    r_squared = r2_score(y, y_pred)
    
    print("=============Linear Regression=============")
    print("Error: %.4f" % sse)
    print("R2: %.4f" % r_squared)
    print(f"Beta: {linear_regression.coef_}")

if method == 4 or method == 0:
    
    values = []
    
    # Alphas values
    start = 0.1
    end = 10000
    step = 0.1
    alphas = np.arange(start, end + step, step)
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(x, y)
        values.append([alpha,ridge.score(x, y)])
    
    with open("ridge.txt", "w") as file:
        for alpha,value in values:
            file.write(f"{alpha}\t{value}\n")

if method == 5 or method == 0:
    
    values = []
    
    # Alphas values
    start = 0.1
    end = 10000
    step = 0.1
    alphas = np.arange(start, end + step, step)
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(x, y)
        values.append([alpha,ridge.score(x, y)])
        
    with open("lasso.txt", "w") as file:
        for alpha,value in values:
            file.write(f"{alpha}\t{value}\n")

if method == 6 or method == 0:
    regression_model = LinearRegression()
    scores = cross_val_score(regression_model, x, y, cv=6, scoring='r2')
    
    print("=============Cross Validation=============")
    print(f"R2: {scores}")
    
if method == 7 or method == 0:
    regression_model = LinearRegression()
    scores = cross_val_score(regression_model, x, y, cv=2, scoring='r2')
    
    print("=============Cross Validation=============")
    print(f"R2: {scores}")
    
if method == 8 or method == 0:
    
    values = []
    
    kf = KFold(n_splits=15)

    # Iterate through the folds and split the data
    for train_indices, validation_indices in kf.split(x):
        X_train, X_validation = x[train_indices], x[validation_indices]
        y_train, y_validation = y[train_indices], y[validation_indices]
        
        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)
        y_pred = linear_regression.predict(x)

        sse = mean_squared_error(y, y_pred) * len(y)
        r_squared = r2_score(y, y_pred)
        values.append(r_squared)
    
    print("=============Cross Validation=============")
    print(f"R2: {values}")
    print(sum(values) / len(values))
        