import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np
from matplotlib.colors import ListedColormap
import os
plt.style.use("fivethirtyeight")


def prepare_data(df):
    """it will segregate the dependent and independent variable

    Args:
        df (pd.dataframe): it will be a pandas dataframe

    Returns:
        tuple: it will return a tuple of dependent and independent variable
    """
    X = df.drop("y", axis=1)

    y = df["y"]

    return X, y


def save_model(model, filename):
    """it will save the model to a new file

    Args:
        model (trained model): it will save the trained model
        filename (string): file where trained model will be saved
    """
    model_dir = "models"
    # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    os.makedirs(model_dir, exist_ok=True)
    filePath = os.path.join(model_dir, filename)  # model/filename
    joblib.dump(model, filePath)


def save_plot(df, file_name, model):
    """it will save the plot
    Args:
        df (pandas dataframe): it a pandas dataframe      file_name (_type_): _description_
        model (string): trained
    """

    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classfier, resolution=0.02):
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(y))])

        X = X.values  # as a array
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        print(xx1)
        print(xx1.ravel())
        Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()

    X, y = prepare_data(df)

    _create_base_plot(df)
    _plot_decision_regions(X, y, model)

    plot_dir = "plots"
    # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    os.makedirs(plot_dir, exist_ok=True)
    plotPath = os.path.join(plot_dir, file_name)  # model/filename
    plt.savefig(plotPath)
