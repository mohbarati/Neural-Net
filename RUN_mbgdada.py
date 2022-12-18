"""
    Run this code for mini-batch Adaline classifier
    on Iris Data (2 columns). It uses mini batch 
    adaline gradient descent (mb_ada). You can 
    change the activation function to any of these:
    linear, binstep (sinary step), sigmoid,
    tanh, ReLU, softplus, silu (x*sigmoid),
    and L_ReLU (leaky ReLU). Also, batch size 
    can be changed. Change learning rate if required.
"""

import numpy as np
import pandas as pd
from mb_ada import NN

# Data pre-processing
df = pd.read_csv("iris.data", header=None, encoding="utf-8").iloc[:100]
fixed_prob = np.random.RandomState(1)  # fixed probability for reproducibility
shuffled_index = fixed_prob.permutation(df.index)

# shuffling the dataset before dividing the test/train portions
df = df.reindex(shuffled_index)
# convert categorical variables into numerical
df[df.columns[4]] = np.where(df.iloc[0:100, 4] == "Iris-setosa", 1, -1)
# extract train info
y = df.iloc[0:80, 4]
x = df.iloc[0:80, :4]
# extract test info
y_test = df.iloc[80:100, 4]
x_test = df.iloc[80:100, :4]
norm = True  # in case Z-score Normalization is needed, False otherwise
if norm:
    x_test = (x_test - np.mean(x, axis=0)) / np.std(x)
batch_size = 32
# Initialize ADAptive LInear NEuron classifier
ppn = NN(eta=0.01, epoch=15, active_method="linear", batch_size=batch_size)
# train the classifier
ppn.fit(x, y, norm=norm)  # plot: animation plot of weight updates

# test the accuracy and performance
prediction = ppn.predict(x_test)
# printing the result of the test
print(ppn.accuracy(prediction, y_test))
print("activation method: ", ppn.active_method)
print(ppn.target_predict_table(prediction, y_test))
# plot of costs
ppn.plot_costs()
# ppn.plot_active_func()
