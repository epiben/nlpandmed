from datetime import datetime
from random import sample
import psycopg2

DSN = f"host=dbserver user={snakemake.config['user']} dbname={snakemake.config['database']}"

if __name__ == "__main__":

    with psycopg2.connect(DSN) as conn:
        q = f"""
            SELECT DISTINCT label
            FROM (
                SELECT UNNEST(labels) AS label 
                FROM {snakemake.config['keras_table']}
            ) x
            GROUP BY label 
                HAVING count(label) >= %s;
        """ 

        with conn.cursor() as cur:
            cur.execute(q, (snakemake.params['min_label_count'], ))
            labels = [x[0] for x in cur] 
    
    single = [x for x in labels if "_" not in x]
    combos = [x for x in labels if "_" in x]

    print(f"No. single labels: {len(single)}")
    print(f"No. combo labels: {len(combos)}")

    # These lines are for development, using a subset of models
    # labels = single + sample(combos, 500)

    for label in labels:
        try: # don't overwrite existing files
            open(snakemake.params["output_dir"] + label, "x").write(label)
        except:
            pass
    