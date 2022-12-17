import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv(r".\iris.data", header=None)
copy_dataset = dataset.copy()
dataset.insert(0, 0, np.ones(dataset.shape[0]), allow_duplicates=True)
dataset.columns = range(dataset.columns.size)
dataset.iloc[:, -1] = (dataset.iloc[:, -1] == "Iris-setosa").astype(int)
test = pd.concat([dataset.loc[sl] for sl in [slice(0, 9), slice(90, 99)]])
train = dataset.loc[slice(10, 89)]
train_shuffled = train.sample(frac=1, random_state=1).reset_index(drop=True)
# initial parameters
n = test.shape[1] - 1
bias = 1
prng = np.random.RandomState(46)
weights = prng.normal(0, 0.1, n)
weights[0] = bias
epoch = 3 * n
etha = 0.0001
# activation function
active = lambda X: np.where(X >= 0, 1, 0)
# incorrect target values for each epoch
errors = []

plt.ion()
# creating subplot and figure
fig = plt.figure()
ax = fig.add_subplot(111)
(line1,) = ax.plot(range(len(weights)), weights)
plt.title("animated plot of weights over itterations", fontsize=20)
plt.xlabel("index for each weight")
plt.ylabel("weights")
ax.set_xlim(0, 4)
ax.set_ylim(-5, 5)

for _ in range(epoch):
    x = train_shuffled.iloc[:, :-1]  # input train data (shuffled)
    y = train_shuffled.iloc[:, -1]  # target values
    z = np.sum(x * weights, axis=1)  # net input
    predicted = active(z)  # perceptron prediction
    loss = etha * (y - z)  # amount of update
    weights += np.dot(x.T, loss)  # update of weights
    error = sum(y - z != 0)  # number of erroneous predictions
    errors.append(error)

    line1.set_xdata(range(len(weights)))
    line1.set_ydata(weights)
    # plt.plot(range(len(weights)), weights)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
plt.ioff()
plt.title("You can Close This Window", fontsize=20)
plt.show()
# check
x_test = test.iloc[:, :-1]  # test data
# predicted output value
predicted = np.round(np.sum(x_test * weights, axis=1))
y_test = test.iloc[:, -1]  # target values
y_test.name = "Target"
discrepency = y_test != predicted
df = pd.DataFrame(y_test[discrepency])
df["predicted"] = predicted[discrepency]
if np.any(discrepency):
    print("{} errors were found: ".format(sum(discrepency)))
    print(df)
else:
    print("No error was found during testing.")

x1 = copy_dataset.iloc[:100, 0]
x2 = copy_dataset.iloc[:100, 1]
x3 = copy_dataset.iloc[:100, 2]
x4 = copy_dataset.iloc[:100, 3]
y = copy_dataset.iloc[:100, -1]
fig, axs = plt.subplots(2, 2)
fig.suptitle("Spacial seperation of feature parameters", fontsize=20)
axs[0, 0].scatter(x1[:50], x2[:50], label="Iris-setosa")
axs[0, 0].scatter(x1[50:], x2[50:], label="Iris-versicolor")
axs[0, 0].legend(loc="upper left")

axs[0, 1].scatter(x1[:50], x3[:50], label="Iris-setosa")
axs[0, 1].scatter(x1[50:], x3[50:], label="Iris-versicolor")
axs[0, 1].legend(loc="upper left")

axs[1, 0].scatter(x1[:50], x4[:50], label="Iris-setosa")
axs[1, 0].scatter(x1[50:], x4[50:], label="Iris-versicolor")
axs[1, 0].legend(loc="upper left")

axs[1, 1].scatter(x2[:50], x3[:50], label="Iris-setosa")
axs[1, 1].scatter(x2[50:], x3[50:], label="Iris-versicolor")
axs[1, 1].legend(loc="upper right")
figManager = plt.get_current_fig_manager()
figManager.window.state("zoomed")
plt.show()

from matplotlib.colors import ListedColormap

# setup marker generator and color map
markers = ("s", "x", "o", "^", "v")
colors = ("red", "blue")
cmap = ListedColormap(colors)


res = 0.01
x1min, x1max = x1.min() - 1, x1.max() + 1
x2min, x2max = x2.min() - 1, x2.max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, res), np.arange(x2min, x2max, res))
length = xx1.ravel().shape
new_df = pd.DataFrame()
new_df[0] = np.ones(length)
new_df[1] = xx1.ravel()
new_df[2] = xx2.ravel()
new_df[3] = np.ones(length) * 5
new_df[4] = np.ones(length) * -5
Z = active(np.sum(new_df * weights, axis=1))
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.show()
