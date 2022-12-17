import numpy as np
import pandas as pd
from adasgd import NN

# Data pre-processing
df = pd.read_csv("iris.data", header=None, encoding="utf-8").iloc[:100]
# fixed_prob = np.random.RandomState(10)
shuffled_index = np.random.permutation(df.index)  # fixed_prob.permutation(...)
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

# Initialize ADAptive LInear NEuron classifier
ppn = NN(eta=0.01, epoch=15, active_method="linear")
# train the classifier
ppn.fit(x, y, norm, plot=False)  # plot: animation plot of weight updates

# test the accuracy and performance
prediction = ppn.predict(x_test)
# printing the result of the test
print(ppn.accuracy(prediction, y_test))
# print(ppn.target_predict_table(prediction, y_test))
# plot of costs
# ppn.plot_costs()