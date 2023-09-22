from itertools import combinations
import numpy as np 

def split_data(x, y, number):
    index_combinations = list(combinations(range(len(x)), number))

    x_test = []
    x_train = []
    y_test = []
    y_train = []

    for indices in index_combinations:
        
        x_test.append(np.array([x[i] for i in indices]))
        x_train.append(np.array([x[i] for i in range(len(x)) if i not in indices]))

        y_test.append(np.array([y[i] for i in indices]))
        y_train.append(np.array([y[i] for i in range(len(y)) if i not in indices]))

    return x_test, x_train, y_test, y_train


"""
# Generate some example data
x = np.array([[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],[2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9],[3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9],[4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9],[5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9],[6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9]])
y = np.array([[1],[2],[3],[4],[5],[6]])    # Replace with your actual vector y
k = 2

x_test, x_train, y_test, y_train = split_data(x, y, k)

for i in range(len(x_test)):
    print(x_test[i])
    print(y_test[i])
    print(x_train[i])
    print(y_train[i])
    print("====================================")



"""