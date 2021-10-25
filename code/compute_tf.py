from psycopg2.sql import SQL, Identifier
from tqdm import tqdm

import datetime
import multiprocessing
import numpy as np
import psycopg2
import sys

config = snakemake.config
DSN = f"host=dbserver user={config['user']} dbname={config['database']}"

class Worker(object):

    def __init__(self):

        self.bth_con = psycopg2.connect(DSN)

    def __call__(self, tasks):

        query = f"""
            BEGIN;

            SET search_path TO {config['schema']};

            WITH cte_txt_by_visit AS (
                SELECT 
                    visit_key,
                    regexp_replace(string_agg(retained_tokens, ' '), '\s+', ' ', 'gi') AS tokens
                FROM {config['preprocess_table']} AS txt
                INNER JOIN adm
                    ON adm.pid = txt.pid
                    AND txt.datetime BETWEEN adm.adm_datetime 
                        AND adm.adm_datetime + '{snakemake.params['window']} hours'::interval
                WHERE visit_key = ANY(%s)
                GROUP BY visit_key
            ),
            cte_tokens AS (
                SELECT
                    visit_key
                    , unnest(string_to_array(tokens, ' ')) AS token
                FROM cte_txt_by_visit
            )
            INSERT INTO {config['tf_table']}
            SELECT 
                visit_key
                , token
                , count(token) as tf
            FROM cte_tokens 
            WHERE token <> ''
            GROUP BY visit_key, token;

            COMMIT;
        """

        with self.bth_con.cursor() as cur:
            cur.execute(query, (tasks, ))
            self.bth_con.commit()

        return None

def subprocess(tasks):
    
    sys.stdout = open(snakemake.log[0], "a")

    worker = Worker()
    for i, chunk in enumerate(np.array_split(tasks, 100)):
        chunk = chunk.tolist()
        worker(chunk)

    print(str(datetime.datetime.now()) + " done")

    sys.stdout.close()

    return None

def child_initialize(_worker, _config, _snakemake):
    
    global Worker, config, snakemake, DSN
    Worker = _worker
    config = _config
    snakemake = _snakemake
    DSN = f"host=dbserver user={config['user']} dbname={config['database']}"

if __name__ == "__main__":

    sys.stdout = open(snakemake.log[0], "a")

    print ("###\tTable Init.. \n")
    con = psycopg2.connect(DSN)
    cur = con.cursor()
    cur.execute(f"""
        BEGIN;
        DROP TABLE IF EXISTS {config['tf_table']} CASCADE;
        CREATE TABLE {config['tf_table']} (visit_key TEXT, token TEXT, tf INT);
        GRANT ALL PRIVILEGES ON {config['tf_table']} TO bth_user;
        COMMIT;
        """)
    con.close()

    print ("###\tPopulate table.. \n")
    con = psycopg2.connect(DSN)

    with con.cursor("cur1") as cur1:

        print("Setting up multiprocessing")
        cur1.itersize = 10000 # only with named cursor
        query = f"""
            SELECT DISTINCT visit_key
            FROM {config['preprocess_table']} AS txt
            INNER JOIN nlpandmed.adm
                ON adm.contact_type = 'I'
                AND adm.pid = txt.pid
                AND txt.datetime BETWEEN adm.adm_datetime 
                    AND adm.adm_datetime + '{snakemake.params["window"]} hours'::interval;
        """
        cur1.execute(query) 
        tasks = [el[0] for el in cur1]
        tasks = np.array_split(tasks, snakemake.threads)
    
    con.commit()
    con.close()

    print ("Start multiprocessing.. ")
    with multiprocessing.Pool(snakemake.threads, initializer=child_initialize, 
        initargs=(Worker, config, snakemake)) as pool:
        pool.map(subprocess, tasks) 

    con = psycopg2.connect(DSN)
    cur = con.cursor()
    
    print(str(datetime.datetime.now()) + " Creating index")
    cur.execute(f"CREATE INDEX ON {config['tf_table']} USING HASH (token);") 

    print(str(datetime.datetime.now()) + " Creating view with distinct visit_keys")
    cur.execute(f"""
        BEGIN;
        SET SEARCH_PATH TO {config['schema']};

        DROP MATERIALIZED VIEW IF EXISTS {config['tf_table']}_visit_keys;

        CREATE MATERIALIZED VIEW {config['tf_table']}_visit_keys AS 
        SELECT DISTINCT visit_key FROM {config['tf_table']};
        
        COMMIT;
    """)

    print(str(datetime.datetime.now()) + " Creating view with distinct tokens")
    cur.execute(f"""
        BEGIN;
        SET SEARCH_PATH TO {config['schema']};

        DROP MATERIALIZED VIEW IF EXISTS {config['tf_table']}_tokens;

        CREATE MATERIALIZED VIEW {config['tf_table']}_tokens AS 
        SELECT DISTINCT token FROM {config['tf_table']};
        
        COMMIT;
    """)

    con.close()

    with open(snakemake.output[0], "w") as f:
        f.write(str(datetime.datetime.now()))

    sys.stdout.close()