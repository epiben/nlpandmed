"""
Utility functions used in multiple places

"""

from datetime import datetime
from tensorflow.keras.models import load_model

import fasttext
import numpy as np
import pandas as pd
import re
import sqlalchemy

def status(message):

    """ Prints a timestamped status message to stdout

    Parameters
    ----------
    message : str
        Message to be printed

    """

    print(f"{datetime.now()}:\033[0;35m {message} \033[0;0m ") 
        # make message purple so it stands out from other stuff


def fetch_keras_models(model_paths):

    """ Create a generator of Keras models to easily query these

    Parameters
    ----------
    model_paths : list of str
        Full paths to Keras models in .hdf5 format.

    Return
    ------
    Generator of (model name, model object) tuples

    """

    names = [x.split("/")[-1].replace(".hdf5", "") for x in model_paths] 
    for model_name, model_path in zip(names, model_paths):
        yield (model_name, load_model(model_path))

def make_exposure_predictions(terms_with_vectors, target_label, drug_model):
    
    """ Make exposure predictions given vectors and models

    Parameters
    ----------
    terms_with_vectors: list of dicts
        Must contain domain, main_term, term and term's word vector
    target_label: str
    drug_model: keras model
        The model whose predict method is used to create profiles.

    Returns
    -------
    Pandas data frame with six columns: target_label, domain, main_term, term,
        pred [predicted probabilities], word_vector [of main_term, as string] 

    """
    
    vectors = [t["word_vector"] for t in terms_with_vectors] 

    # Ensure well-behaved predictions if single vector
    if len(vectors) == 1:
        v = np.expand_dims(vectors[0], axis=0)
    else:
        v = np.array(vectors)

    # FIX: vectors might need to be tensors to avoid redundant retracing

    # sqlalchemy doesn't seem to play well with array-columns in postgresql
    vectors_as_strings = [ 
        "; ".join(str(x) for x in vector) for vector in vectors
    ]
    
    return pd.DataFrame({ 
        "target_label": target_label,
        "domain": [t["domain"] for t in terms_with_vectors],
        "main_term": [t["main_term"] for t in terms_with_vectors],
        "term": [t["term"] for t in terms_with_vectors],
        "pred": np.squeeze(drug_model.predict(v)),
        "word_vector": vectors_as_strings
    })

def save_table_in_db(df, full_table_name, engine, if_exists="replace"):

    """ Saves the data frame in the given table

    Parameters
    ----------
    df: Pandas data frame
    full_table_name: str
        Full table name with period separating schema and actual table name
    engine: sqlalchemy engine
        Result of sqlalchemy.create_engine()
    if_exists: str, default="replace"
        What to do if table already exists. See pd.to_sql() for details.

    Returns
    -------
    True if successful, False if not. 
    
    """

    try:
        schema, table = full_table_name.split(".")
        df.to_sql(table, engine, schema, if_exists, index=False)
        return True
    except:
        return False
    