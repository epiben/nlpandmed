from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from psycopg2.sql import SQL, Identifier
from tqdm import tqdm

import datetime
import multiprocessing
import numpy as np
import psycopg2
import sys

DSN = f"host=dbserver user={snakemake.config['user']} dbname={snakemake.config['database']}"

def status(message):

    print(str(datetime.datetime.now()) + f" {message}")

def subprocess(tasks, n_docs):
    
    sys.stdout = open(snakemake.log[0], "a")

    status("Starting subprocess")
    query = f"""
        BEGIN;
        INSERT INTO {snakemake.config['tfidf_table']}
        SELECT 
            *
            , tf * log({n_docs}/(1.0 + df)) AS tfidf
        FROM (
            SELECT 
                visit_key
                , token
                , tf
                , count(visit_key) OVER(PARTITION BY token) AS df
            FROM {snakemake.config['tf_table']}
            WHERE token = ANY(%s)
        ) AS x
        WHERE df >= {snakemake.params['min_df']};
        COMMIT;
    """
    
    status("Running computation on server")
    conn = psycopg2.connect(DSN)
    with conn.cursor() as cur:
        cur.execute(query, (tasks.tolist(), ))
    conn.close()

    status("Done")

    sys.stdout.close()
    
    return None

def create_index(spec):

    sys.stdout = open(snakemake.log[0], "a")

    query = f"CREATE INDEX ON {snakemake.config['tfidf_table']} {spec};"
    status("Creating index: " + query)

    conn = psycopg2.connect(DSN)
    with conn.cursor() as cur:
        cur.execute(query)
    conn.commit()
    conn.close()

    status("Created index: " + query)

    sys.stdout.close()

    return None

if __name__ == "__main__":

    sys.stdout = open(snakemake.log[0], "a")

    conn = psycopg2.connect(DSN)

    status("Prepping database")
    with conn.cursor() as cur:
        q = f"""
            BEGIN;
            DROP TABLE IF EXISTS {snakemake.config['tfidf_table']};
            CREATE TABLE {snakemake.config['tfidf_table']} (
                visit_key TEXT, token TEXT, tf INT, df INT, tfidf FLOAT
            );
            GRANT ALL PRIVILEGES ON {snakemake.config['tfidf_table']} TO bth_user;
            COMMIT;
        """
        cur.execute(q)
    conn.close()

    conn = psycopg2.connect(DSN)

    status("Counting number of documents")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT count(DISTINCT visit_key) 
            FROM {snakemake.config["tf_table"]};
        """)
        n_docs = cur.fetchone()[0] 
    
    with conn.cursor("cur1") as cur1:
        status("Setting up multiprocessing")
        cur1.itersize = 10000  # only with named cursor
        q = f"""SELECT DISTINCT token FROM {snakemake.config["tf_table"]};"""
        cur1.execute(q) 
        tasks = [el[0] for el in cur1]
        print(f"Found all {len(tasks)} tokens, making {snakemake.threads} tasks")
        tasks = np.array_split(tasks, snakemake.threads)
    
    conn.commit()
    conn.close()

    status("Starting multiprocessing")
    with ProcessPoolExecutor(max_workers=snakemake.threads) as executor:
        executor.map(subprocess, tasks, repeat(n_docs)) 

    status("Creating indices")

    index_specs = ["(token)", "(tfidf DESC)", "USING HASH (visit_key)"]
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.map(create_index, index_specs)

    # Write output file monitored by snakemake
    open(snakemake.output[0], "w").write(str(datetime.datetime.now()))

    sys.stdout.close()
