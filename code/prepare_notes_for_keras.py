from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from psycopg2.sql import SQL, Identifier
from tqdm import tqdm

import datetime
import io
import itertools
import json
import multiprocessing
import nltk
import numpy as np
import pandas as pd
import psycopg2
import re
import string
import sys

SEP = '\t'
config = snakemake.config
DSN = f"host=dbserver user={config['user']} dbname={config['database']}"
nltk.data.path.append('/services/tools/anaconda3/4.0.0/data/nltk_data/')

def status(message):
    print(str(datetime.datetime.now()) + f": {message}")

class PseudoFile(io.TextIOBase):
    """ given an iterator which yields strings,
    return a file like object for reading those strings """

    def __init__(self, it):
        self._it = it
        self._f = io.StringIO()

    def read(self, length=sys.maxsize):

        try:
            while self._f.tell() < length:
                self._f.write(next(self._it) + "\n")

        except StopIteration as e:
            pass

        finally:
            self._f.seek(0)
            data = self._f.read(length)

            # save the remainder for next read
            remainder = self._f.read()
            self._f.seek(0)
            self._f.truncate(0)
            self._f.write(remainder)
            return data

    def readline(self):
        return next(self._it)


class Worker(object):

    def __init__(self):

        self.bth_con = psycopg2.connect(DSN)
        self.dest_table = config['preprocess_table']
        self.negation_window_range = range(snakemake.params['negation_window'] + 1)
        self.min_token_length = snakemake.params['min_token_length']
        self.regex_punctuation = "(?<!\d)[^\w\s](?!\d)|(?<=\d)[^\w\s](?!\d)|(?<!\d)[^\w\s](?=\d)|(?<=\d)[^\w\s.,](?=\d)"
        self.negations = {'ikke', 'ingen', "ej", "heller", "hverken", "kendt", "tidligere", "obs"}

    def extract(self, task):

        pid, rekvdt, txt_id, raw_text = task  # unpack
        text = self.remove_tail(raw_text)
        retained, negated = self.remove_negated_words__punctuation__whitespace__lower(text)
        retained = self.remove_names(retained)
        retained = self.remove_stopwords(retained)

        datetime, _ = self.parse_date(rekvdt)
        row = SEP.join((txt_id, str(pid), datetime, " ".join(retained), " ".join(negated)))
        yield row

    def parse_date(self, date_str):
        
        if date_str == 'nan' or date_str == 'NA':
            date_str = END_OF_DATA
        if len(date_str) == 10:
            format_str = '%Y-%m-%d'
        elif len(date_str) == 4:
            format_str = '%Y'
        elif len(date_str) == 12:
            format_str = '%Y%m%d%H%M'
        elif len(date_str) == 20:
            format_str = "%Y-%m-%dT%H:%M:%SZ"
        else:
            format_str = '%Y-%m-%d %H:%M'
            
        dt = datetime.datetime.strptime(date_str, format_str)
        timestamp = datetime.datetime.timestamp(dt)
        return str(dt), str(timestamp)

    def clean_csv_value(self, value):
        
        if value is None:
            return r'\N'
        value = str(value).replace(';', ',')
        value = value.replace('\\', '')
        return value

    def remove_tail(self, text):
        
        regex_tail = "\.[^.]{2,50}/.{3,30}$"
        text_no_tail = re.sub(regex_tail, "", text)
        return text_no_tail

    def remove_negated_words__punctuation__whitespace__lower(self, text):
        
        text = re.sub("\s+", " ", text)

        retained_tokens = []
        negated_tokens = []
        
        for sent in sent_tokenize(text, language = "danish"):
            words = word_tokenize(sent, language = "danish")
            words = [re.sub(self.regex_punctuation, "", w).lower() for w in words]

            negated_idx = [i+j for i,x in enumerate(words) for j in self.negation_window_range if x in self.negations]
            retained_tokens.extend(w for k,w in enumerate(words) if k not in negated_idx and len(w) >= self.min_token_length)
            negated_tokens.extend(w for k,w in enumerate(words) if k in negated_idx and k < len(words))

        return retained_tokens, negated_tokens
        
    def remove_stopwords(self, tokens):

        tokens_wo_stopwords = [t for t in tokens if t not in danish_stop_words]
        return tokens_wo_stopwords

    def remove_names(self, tokens):
        
        tokens_wo_names = [t for t in tokens if t not in names]
        return tokens_wo_names

    def __call__(self, tasks):
        
        q = """
            SELECT pid, rekvdt, txt_id, text_t1 
            FROM epj_210302.txt 
            WHERE pid = ANY(%s) 
                AND text_t1 <> '';
        """

        with self.bth_con.cursor() as cur1:
            cur1.execute(q, (tasks, )) 

            results = (row for task in cur1 for row in self.extract(task))
            output = PseudoFile(results)

            with self.bth_con.cursor() as cur2:
                cur2.copy_from(output, self.dest_table, sep=SEP)

        self.bth_con.commit()

        return None

def subprocess(tasks):

    sys.stdout = open(snakemake.log[0], "a")

    status("Starting child process")

    worker = Worker()
    for chunk in np.array_split(tasks, 1000):
        worker(chunk.tolist())
    
    status("Done")

    sys.stdoutclose()

    return None

def child_initialize(_worker, _names, _stemmer, _tokenizer, _danish_stop_words, _config, _snakemake):
    global Worker, names, stemmer, tokenizer, danish_stop_words, config, snakemake, DSN
    Worker = _worker
    names = _names
    stemmer = _stemmer
    tokenizer = _tokenizer
    danish_stop_words = _danish_stop_words
    config = _config
    snakemake = _snakemake
    DSN = f"host=dbserver user={config['user']} dbname={config['database']}"

if __name__ == "__main__":
    
    sys.stdout = open(snakemake.log[0], "a")

    status("Loading stopwords")
    danish_stop_words = set(stopwords.words('danish'))

    status("Loading names")
    names = pd.read_csv(
        open(snakemake.input["names"]).readline(), 
        usecols=[0], squeeze=True, sep='\t', error_bad_lines=False, header=None
    )
    names = {x.lower() for x in names.map(str)}

    status("Prepping database")
    con = psycopg2.connect(DSN)
    cur = con.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {config['preprocess_table']} CASCADE;")
    con.commit()

    create_table_query = f"""
        CREATE TABLE {config['preprocess_table']} (
            txt_id TEXT, 
            pid INT, 
            datetime TIMESTAMP, 
            retained_tokens TEXT, 
            negated_tokens TEXT
        );
        GRANT ALL PRIVILEGES ON {config['preprocess_table']} TO bth_user;
        """

    cur.execute(create_table_query)
    con.commit()
    con.close()

    status("Populate table")
    con = psycopg2.connect(DSN)

    with con.cursor("cur1") as cur1:

        print("Setting up multiprocessing")
        n_tasks = snakemake.threads
        cur1.itersize = 10000  # only with named cursor
        cur1.execute("""
            SELECT DISTINCT pid 
            FROM epj_210302.txt
            LIMIT 5000000;
        """) 
        tasks = [el[0] for el in cur1]
        tasks = np.array_split(tasks, n_tasks)

        print ("Start multiprocessing.. ")
        with multiprocessing.Pool(n_tasks, initializer=child_initialize, 
            initargs=(Worker, names, sent_tokenize, word_tokenize, danish_stop_words, config, snakemake)) as pool:
            pool.map(subprocess, tasks) 
    
    con.commit()
    con.close()

    con = psycopg2.connect(DSN)
    
    cur = con.cursor()
    print(str(datetime.datetime.now()) + " Creating indices")
    cur.execute(f"""
        BEGIN; 
        
        CREATE INDEX ON {config['preprocess_table']} USING HASH (pid); 
        CREATE INDEX ON {config['preprocess_table']} (datetime); 
        
        COMMIT;
    """) 
    con.close()

    with open(snakemake.output[0], "w") as f:
        f.write(str(datetime.datetime.now()))

    sys.stdout.close()