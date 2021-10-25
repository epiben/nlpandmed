"""
Create the X and y for tf-keras model. Assumes the following structure of the data file:

    # start of file
    __label__[ATC] __label__[ATC] token1 token2 token3
    __label__[ATC] token1 token2 token3 token4 token5
    # end of file
"""

from itertools import repeat
from tqdm import tqdm

import fasttext
import multiprocessing
import numpy as np
import scipy.sparse as sparse
import subprocess

def extract_labels(data_file):

    """
    Make a set of unique labels. Assumes that labels are prefixed by "__label__". 
    """

    labels = set()
    length, _ = subprocess.check_output(f"wc -l {data_file}", shell=True).decode().split()
    
    with open(data_file, "r") as f:
        for line in tqdm(f, total=int(length)):
            for word in line.split():
                if "__label__" in word:
                    labels.add(word.replace("__label__", ""))
                else:
                    break

    return list(labels)

def extract_xy(obs, vector_dict, labels):

    """
    obs: list of 2-element tuples (first is labels, second is tokens)
    vector_dict: dict with embedding vectors for all tokens in training set
    labels: list of all labels for one-hot-encoding labels in obs
    """

    x, y = [], []

    for row in obs:
        row_labels = np.array([1 if l in row[0].split() else 0 for l in labels])
        vectors = [vector_dict[t] for t in row[1].split()] 
    
        y.extend(repeat(row_labels, len(vectors)))
        x.extend(vectors)

    return np.stack(x), np.stack(y) #sparse.coo_matrix() to reduce memory

def child_initialize(_model, _labels, _pbar):

    global model, labels, pbar
    model = _model
    labels = _labels
    pbar = _pbar

def chunkify(lst, n):

    for i in range(0,len(lst), n):
        yield lst[i:i+n]

if __name__ == "__main__":

    input_data = snakemake.input["data"]
    labels = extract_labels(input_data)
    model = fasttext.load_model(snakemake.input["embedding_model"])
    length, _ = subprocess.check_output(f"wc -l {input_data}", shell=True).decode().split()
    print(f"Num of records: {length}")
    
    with open(input_data, "r") as f:
        lines = f.readlines()

    # this x, y can be directly used as input of a Keras model    
    x, y  = extract_xy(lines[:1000000], model, labels)
