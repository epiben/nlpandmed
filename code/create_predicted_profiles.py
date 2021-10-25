"""
Generate predicted medication profiles, collecting result of all trained
Keras models.

"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
from utils import \
    fetch_keras_models, make_exposure_predictions, save_table_in_db, status

import fasttext
import json
import numpy as np
import os
import pandas as pd
import psycopg2
import re
import tqdm
import yaml


if 'snakemake' not in globals():
    import interactive
    print("## Warning: loading a custom snakemake object ##")
    snakemake = interactive.SnakeMakeClass()
else:
    sys.stdout = open(snakemake.log[0], "a")
    sys.stderr = open(snakemake.log[0], "a")

DSN = f"host=dbserver user={snakemake.config['user']} dbname={snakemake.config['database']}"

def workhorse(model_paths):

    """ Workhorse of the script, can be run parallelly

    Parameters
    ----------
    model_paths : list of str
        Full paths to Keras models in .hdf5 format.

    Return
    ------
    Saves result in database, invisibly.

    """

    sys.stdout = open(snakemake.log[0], "a")
    sys.stderr = open(snakemake.log[0], "a")
    
    workhorse_id = hash(" ".join(model_paths))
    
    status(f"Starting workhorse {workhorse_id}")

    status("Setting up terms")
    with open(snakemake.input["terms"], "r") as f:
        terms_dict = yaml.safe_load(f)
    
    status("Loading fasttext model")
    fasttext_model = fasttext.load_model(snakemake.input["embedding_model"])

    terms_with_vectors = [] 
    for domain, term_list in terms_dict.items(): 
        for main_term, synonyms in term_list.items(): 
            for term in [main_term] + synonyms:
                terms_with_vectors.append({
                    "domain": domain, 
                    "main_term": main_term, 
                    "term": term, 
                    "word_vector": fasttext_model.get_word_vector(term)
                })
    
    status("Making df with predicted profiles")
    keras_models = fetch_keras_models(model_paths) # generator
    preds = []
    for label, model in keras_models:
        preds.append(make_exposure_predictions(terms_with_vectors, label, model))

    status("Writing df to database")
    config = snakemake.config
    psql_url = f"""postgresql://{config["user"]}@dbserver/{config["database"]}"""
    engine = create_engine(psql_url)

    save_table_in_db(
        df=pd.concat(preds), 
        full_table_name=config["predictions_table"], 
        engine=engine, 
        if_exists="append"
    )

    status(f"Workhorse {workhorse_id} done")

    return None

if __name__ == "__main__":

    config = snakemake.config
    params = snakemake.params

    status("Prepping database")
    conn = psycopg2.connect(DSN)
    with conn.cursor() as cur:
        q = f"""
            BEGIN;
            DROP TABLE IF EXISTS {config['predictions_table']} CASCADE;
            CREATE TABLE {config['predictions_table']} (
                target_label TEXT
                , domain TEXT
                , main_term TEXT
                , term TEXT
                , pred FLOAT
                , word_vector TEXT
            );
            GRANT ALL PRIVILEGES ON {config['predictions_table']} TO bth_user;
            COMMIT;
        """
        cur.execute(q)
    conn.close()

    status("Setting up and running in parallel")
    model_paths = snakemake.input["keras_models"]
    tasks = np.array_split(model_paths, snakemake.threads)
    tasks = [task.tolist() for task in tasks] 

    with ProcessPoolExecutor(max_workers=snakemake.threads) as executor:
        executor.map(workhorse, tasks)

    status("Creating ranked signals table in database")
    conn = psycopg2.connect(DSN)
    with conn.cursor() as cur:
        cur.execute(f"""
            BEGIN;

            DROP TABLE IF EXISTS {config["signals_table"]} CASCADE;

            WITH cte_label_frequencies AS (
                SELECT target_label, freq
                FROM {config["label_frequencies_table"]}
            ),
            cte_pertinent_models AS (
                SELECT target_label
                FROM {config["eval_table"]}
                WHERE auroc >= {params["min_auroc"]} 
                    AND intercept BETWEEN {params["min_intercept"]} 
                        AND {params["max_intercept"]} 
                    AND slope BETWEEN {params["min_slope"]}
                        AND {params["max_slope"]} 
            ),
            cte_signals_with_oddsratios AS (
                SELECT 
                    domain
                    , main_term
                    , term
                    , target_label
                    , (pred / (1 - pred)) / (freq / (1 - freq)) AS odds_ratio
                FROM {config["predictions_table"]} 
                INNER JOIN cte_label_frequencies USING(target_label)
                INNER JOIN cte_pertinent_models USING(target_label)
            )
            SELECT 
                *
                , DENSE_RANK() OVER( 
                    PARTITION BY domain, main_term, term 
                    ORDER BY odds_ratio DESC
                    ) AS signal_rank
            INTO {config["signals_table"]}
            FROM cte_signals_with_oddsratios;

            GRANT ALL PRIVILEGES ON {config["signals_table"]} TO bth_user;

            CREATE INDEX ON {config["signals_table"]} (signal_rank);
            
            COMMIT;
        """)
    conn.close()

    # Write output monitored by snakemake
    open(snakemake.output[0], "w").write(str(datetime.now()))

    status("Done")
