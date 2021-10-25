from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import datetime
import numpy as np
import psycopg2
import sys

DSN = f"host=dbserver user={snakemake.config['user']} dbname={snakemake.config['database']}"

def status(message):
    print(str(datetime.datetime.now()) + f" {message}")

def create_label_tokens_pairs(tasks):
    
    sys.stdout = open(snakemake.log[0], "a")

    status("Starting subprocess")
    query = f"""
        BEGIN;
        
        -- FIRST LABELS
        WITH cte_adm_med AS (
            SELECT 
                visit_key
                , prim_atc AS atc
            FROM {snakemake.config['adm_med_table']}
            WHERE visit_key = ANY(%s)
        ),
        cte_labels_long AS (
            SELECT 
                a.visit_key
                , array_to_string(
                    array(SELECT DISTINCT unnest(array[a.atc, b.atc])), 
                    '_'
                ) AS label
            FROM cte_adm_med AS a
                , cte_adm_med AS b
            WHERE a.visit_key = b.visit_key
                AND a.atc <= b.atc -- avoid duplicates in reversed order
        ),
        cte_labels AS (
            SELECT 
                visit_key
                , array_agg(label ORDER BY label) AS labels
            FROM cte_labels_long
            WHERE label ~ '\w\d\d\w\w'
            GROUP BY visit_key
        ),
        --
        -- THEN TOKENS
        cte_tokens_long AS (
            SELECT 
                visit_key
                , token
                , row_number() OVER(PARTITION BY visit_key ORDER BY tfidf DESC) as rn
            FROM {snakemake.config['tfidf_table']}
            WHERE visit_key = ANY(%s)
                AND df <= {snakemake.params['max_df']} 
                AND df >= {snakemake.params['min_df']}
        ),
        cte_tokens AS (
            SELECT 
                visit_key
                , array_agg(token ORDER BY rn) AS tokens
            FROM cte_tokens_long
            -- WHERE rn <= {snakemake.params['n_tokens_per_visit']}
            GROUP BY visit_key
        )
        --
        -- FINALLY COMBINE LABELS AND TOKENS
        INSERT INTO {snakemake.config['keras_table']}
        SELECT 
            COALESCE(cte_labels.visit_key, cte_tokens.visit_key) AS visit_key
            , labels
            , tokens
        FROM cte_labels
        FULL JOIN cte_tokens
            ON cte_tokens.visit_key = cte_labels.visit_key;

        COMMIT;
    """
    
    status("Running computation on server")
    conn = psycopg2.connect(DSN)
    with conn.cursor() as cur:
        cur.execute(query, (tasks.tolist(), tasks.tolist(), ))
    conn.close()

    status("Done")

    sys.stdout.close()
    
    return None

def create_index(spec):

    sys.stdout = open(snakemake.log[0], "a")

    query = f"CREATE INDEX ON {snakemake.config['keras_table']} {spec};"
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
            DROP TABLE IF EXISTS {snakemake.config['keras_table']} CASCADE;
            CREATE TABLE {snakemake.config['keras_table']} (
                visit_key TEXT, 
                labels TEXT[], -- 1D array of TEXT
                tokens TEXT[] -- idem
            );
            GRANT ALL PRIVILEGES ON {snakemake.config['keras_table']} TO bth_user;
            COMMIT;
        """
        cur.execute(q)
    conn.close()

    conn = psycopg2.connect(DSN)

    status("Setting up multiprocessing")
    with conn.cursor("cur1") as cur1:
        cur1.itersize = 10000  # only with named cursor
        cur1.execute(f"""
            SELECT DISTINCT visit_key FROM {snakemake.config["tfidf_table"]};
        """) 
        tasks = [x[0] for x in cur1]
        print(f"Making {snakemake.threads} tasks from {len(tasks)} visit_keys")
        tasks = np.array_split(tasks, snakemake.threads)
    
    conn.commit()
    conn.close()

    status("Starting multiprocessing")
    with ProcessPoolExecutor(max_workers=snakemake.threads) as executor:
        executor.map(create_label_tokens_pairs, tasks)

    status("Creating indices")
    index_specs = ["(visit_key)", "USING GIN (labels)", "USING GIN (tokens)"]
    with ProcessPoolExecutor(max_workers=len(index_specs)) as executor:
        executor.map(create_index, index_specs)

    # Write output monitored by snakemake
    open(snakemake.output[0], "w").write(str(datetime.datetime.now()))
    