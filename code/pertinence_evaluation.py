"""
Evaluate discrimination and calibration of trained model to determine which output nodes to use.
"""

from build_array import extract_labels, extract_xy

from datetime import datetime
from sklearn.calibration import calibration_curve
from tensorflow.keras.models import load_model

import fasttext
import numpy as np

if __name__ == "__main__":

    params = snakemake.params

    mlp = load_model(snakemake.input["model"])

    data = open(snakemake.input["data"], 'r').readlines()
    data = [x for x in train if re.search("==EMPTY==", x) is None] # Ben: FIX: This should probably be handled in a separate data processing step and not here

    fasttext_model = fasttext.load_model(snakemake.input["embedding_model"])

    preds = mlp.predict(
        extract_batch(data, fasttext_model, labels, params["batch_size"]),
        verbose=params["verbose"]
    )
