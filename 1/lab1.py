
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import math
from itertools import combinations
import time
import matplotlib.pyplot as plt
import random
seed = 1
random.seed(seed)
from data_spliter import split_data

x = np.load("X_train_regression1.npy")
y = np.load("y_train_regression1.npy")

method = 9

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
        
if method == 9 or method == 0:
    
    values = []
    subset_sizes = [3]
    
    # Alphas values
    start = 0.01
    end = 1000
    step = 0.01
    alphas = np.arange(start, end + step, step)
    
    for subset_size in subset_sizes:
        x_test, x_train, y_test, y_train = split_data(x, y, subset_size)
    
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            
            values2 = []
            for i in range(len(x_test)):
                ridge.fit(x_train[i], y_train[i])
                y_pred = ridge.predict(x_test[i])**2
                values2.append([r2_score(y_test[i], y_pred),np.sum((y_test[i] - y_pred)**2)])
            
            
            values.append([alpha,subset_size,sum(values2[:][0]) / len(values2[:][0]), sum(values2[:][1]) / len(values2[:][1])])
                
    
    with open("ridge_crossV.txt", "w") as file:
        for alpha,subset_size,value1,value2 in values:
            file.write(f"{alpha}\t{subset_size}\t{value1}\t{value2}\n")

if method == 10:

    fig, axs = plt.subplots(2, 5, figsize=(12, 6))

    # Plot data on each subplot
    axs[0, 0].scatter([row[0] for row in x], y.T[0])
    axs[0, 0].set_title('1')

    axs[0, 1].scatter([row[1] for row in x], y.T[0])
    axs[0, 1].set_title('2')

    axs[0, 2].scatter([row[2] for row in x], y.T[0])
    axs[0, 2].set_title('3')

    axs[0, 3].scatter([row[3] for row in x], y.T[0])
    axs[0, 3].set_title('4')

    axs[0, 4].scatter([row[4] for row in x], y.T[0])
    axs[0, 4].set_title('5')

    axs[1, 0].scatter([row[5] for row in x], y.T[0])
    axs[1, 0].set_title('6')

    axs[1, 1].scatter([row[6] for row in x], y.T[0])
    axs[1, 1].set_title('7')

    axs[1, 2].scatter([row[7] for row in x], y.T[0])
    axs[1, 2].set_title('8')
    
    axs[1, 3].scatter([row[8] for row in x], y.T[0])
    axs[1, 3].set_title('9')
    
    axs[1, 4].scatter([row[9] for row in x], y.T[0])
    axs[1, 4].set_title('10')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

if method == 11 or method == 0:
    
    values = []
    subset_sizes = [3]
    
    # Alphas values
    start = 0.01
    end = 1000
    step = 0.01
    alphas = np.arange(start, end + step, step)
    
    for subset_size in subset_sizes:
        x_test, x_train, y_test, y_train = split_data(x, y, subset_size)
    
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            
            values2 = []
            for i in range(len(x_test)):
                ridge.fit(x_train[i], y_train[i])
                y_pred = ridge.predict(x_test[i])**2
                values2.append([r2_score(y_test[i], y_pred),np.sum((y_test[i] - y_pred)**2)])
            
            
            values.append([alpha,subset_size,sum(values2[:][0]) / len(values2[:][0]), sum(values2[:][1]) / len(values2[:][1])])
                
    
    with open("ridge_crossV.txt", "w") as file:
        for alpha,subset_size,value1,value2 in values:
            file.write(f"{alpha}\t{subset_size}\t{value1}\t{value2}\n")