"""author:premsai
    email:premsaishoo@gmail.com
    """

from oneNeuron.perceptron import Perceptron
import pandas as pd
from utils.all_utils import prepare_data, save_model, save_plot
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]%(message)s"

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # now we have created this log direactory
logging.basicConfig(filename=os.path.join(
    log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")


def main(data, eta, epochs, filename, plotFileName):

    df = pd.DataFrame(data)

    logging.info(f"this is the actual dataframe:{df}")

    X, y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plotFileName, model)


if __name__ == "__main__":

    OR = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 1, 1, 1],
    }

    ETA = 0.3  # 0 and 1
    EPOCHS = 100
    try:
        logging.info("<<<<< starting training>>>>>")
        main(data=OR, eta=ETA, epochs=EPOCHS,
             filename="or.model", plotFileName="or.png")

        logging.info("<<<<training successfully done>>>>\n")

    except Exception as e:
        logging.exception(e)
