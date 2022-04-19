from utils.model import Perceptron
import pandas as pd
from utils.all_utils import prepare_data, save_model, save_plot
import numpy as np


def main(data, eta, epochs, filename, plotFileName):

    df = pd.DataFrame(data)

    print(df)

    X, y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plotFileName, model)


if __name__ == "__main__":

    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 1, 1, 1],
    }

    ETA = 0.3  # 0 and 1
    EPOCHS = 10
    main(data=AND, eta=ETA, epochs=EPOCHS,
         filename="or.model", plotFileName="or.png")
