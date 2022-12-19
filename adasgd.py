import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NN:
    """(Mini-Batch/Stochastic) Gradient Descent ADAptive LInear NEuron classifier.

    Attributes
    -----------
    w : 1d-array
      Weights after fitting.
    cost : list
      Sum-of-squares cost function value in each epoch.

    """

    def __init__(self, eta, epoch, active_method, rand_state=1) -> None:
        """Initialize the object

        Args:
            eta (float): Learning rate (between 0.0 and 1.0)
            epoch (int): Max number of itterations over dataset
            rand_state (int, optional): Random number generator seed for random weight initialization.
            active_method (str): activation function to be used
        """
        self.eta = eta
        self.rand_state = rand_state
        self.epoch = epoch
        self.active_method = active_method

    def activation(self, x, param):
        """Computes the activation"""
        match param:
            case "linear":
                return x
            case "binstep":
                return np.where(x >= 0.5, 1, 0)
            case "sigmoid":
                return 1 / (1 + np.exp(-x))
            case "tanh":
                return np.tanh(x)
            case "ReLU":
                return np.maximum(0, x)
            case "softplus":
                return np.log(1 + np.exp(x))
            case "silu":
                return x / (1 + np.exp(-x))
            case "L_ReLU":
                return np.maximum(0.2 * x, x)
            case _:
                raise Exception("invalid activation function")

    def fit(self, x, y, norm=False, plot=False):
        """Fit training data.
        Args:

            x (array): shape = [n_examples, n_features]
                       Training vectors, where n_examples is
                       the number of examples and n_features is
                       the number of features.
            y (array): shape = [n_examples] ==> Target values.
            norm (bool, optional): pre-normalize data. Defaults to False.
            plot (bool, optional): weight update animation. Defaults to False.
        """
        if norm:
            x = self.normalize(x)
        fixed_prob = np.random.RandomState(self.rand_state)
        self.w = fixed_prob.normal(loc=0, scale=0.1, size=x.shape[1])
        self.b = 1.0
        self.costs = []
        if plot:
            line1, fig = self.plotting()
        for _ in range(self.epoch):
            shuffled_index = np.random.permutation(x.index)
            x = x.reindex(shuffled_index)
            y = y.reindex(shuffled_index)
            error = y - self.predict(x)
            self.w += self.eta * pd.Series(np.matmul(x.T, error))
            self.b += self.eta * error.sum()
            cost = (error**2).sum() / 2.0
            self.costs.append(cost)
            if plot:
                line1.set_xdata(range(len(self.w)))
                line1.set_ydata(self.w)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)
        if plot:
            plt.ioff()
            plt.title("You can Close This Window", fontsize=20)
            plt.show()

    def net_input(self, x):
        """Calculate net input"""
        return np.dot(x, self.w) + self.b

    def predict(self, x, decimal_number=2):
        """Return class label (float)"""
        return np.around(
            self.activation(self.net_input(x), self.active_method), decimal_number
        )

    def normalize(self, x):
        """Standardization (Z-score Normalization) to
        have zero mean & unit variance"""
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    def plotting(self):
        """initializing the plot for further use in fit function
           if plot is set to True.

        Returns:
            object: returns initial frame for the animation
        """
        plt.ion()
        # creating subplot and figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        (line1,) = ax.plot(range(len(self.w)), self.w)
        plt.title("animated plot of weights over itterations", fontsize=20)
        plt.xlabel("index for each weight")
        plt.ylabel("weights")
        plt.grid(True)
        ax.set_xlim(0, 4)
        ax.set_ylim(-2, 2)
        return line1, fig

    def accuracy(self, prediction, target):
        """percentage accuracy

        Args:
            prediction (array): prediction done by adaline
            target (array): target values given

        Returns:
            str: Returns a String with percentage accuracy
        """
        # dividing the prediction into two classes >0 & <0
        binary_adjusted_prediction = self.activation(prediction, "binstep")
        accuracy_perc = (
            100
            - (len(target) - sum(binary_adjusted_prediction == target))
            / len(target)
            * 100
        )
        return f"Accuracy percentage: {accuracy_perc}%"

    def target_predict_table(self, prediction, target):
        """Table consisting of target and corresponding predictions

        Args:
            prediction (array): prediction done by adaline
            target (array): target values given

        Returns:
            Pandas Dataframe: Table with two columns
        """
        prediction = self.activation(prediction, "binstep")
        d = {
            "Test_Targets": target,
            "Predict_by_AdalineSGD": prediction,
        }
        return pd.DataFrame(data=d)

    def plot_costs(self):
        fig = plt.figure(figsize=(15, 10))
        plt.plot(range(1, self.epoch + 1), self.costs)
        plt.xlabel("Epoch no.")
        plt.ylabel("Sum of Errors Squared")
        plt.grid(True)
        plt.xticks(range(1, self.epoch + 1))
        plt.title("Costs Versus Epochs", fontsize=20)
        figManager = plt.get_current_fig_manager()
        figManager.window.state("zoomed")
        plt.show()

    def plot_active_func(self):
        """Plotting all defined activations functions"""
        x = np.linspace(-3, 3, 200)
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].plot(x, self.activation(x, param="linear"), "r")  # row=0, col=0
        ax[0, 0].set_title("Linear activation function")
        ax[0, 0].grid(True)

        ax[0, 1].plot(x, self.activation(x, param="binstep"), "b")  # row=0, col=1
        ax[0, 1].set_title("Binary activation function")
        ax[0, 1].grid(True)

        ax[1, 0].plot(x, self.activation(x, param="sigmoid"), "g")  # row=1, col=0
        ax[1, 0].set_title("Sigmoid activation function")
        ax[1, 0].grid(True)

        ax[1, 1].plot(x, self.activation(x, param="tanh"), "k")  # row=1, col=1
        ax[1, 1].set_title("Tanh activation function")
        ax[1, 1].grid(True)

        ax[2, 0].plot(x, self.activation(x, param="ReLU"), "r--")  # row=2, col=0
        ax[2, 0].set_title("ReLU activation function")
        ax[2, 0].grid(True)

        ax[2, 1].plot(x, self.activation(x, param="softplus"), "b--")  # row=2, col=1
        ax[2, 1].set_title("softplus activation function")
        ax[2, 1].grid(True)

        ax[3, 0].plot(x, self.activation(x, param="silu"), "g--")  # row=3, col=0
        ax[3, 0].set_title("silu activation function")
        ax[3, 0].grid(True)

        ax[3, 1].plot(x, self.activation(x, param="L_ReLU"), "k--")  # row=3, col=1
        ax[3, 1].set_title("Leaky ReLU activation function")
        ax[3, 1].grid(True)
        fig.tight_layout()
        plt.show()
