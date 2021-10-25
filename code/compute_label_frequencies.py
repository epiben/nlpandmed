from datetime import datetime
from utils import status

import psycopg2

config = snakemake.config

if __name__ == "__main__":

	DSN = f"host=dbserver user={config['user']} dbname={config['database']}"

	q = f"""
		BEGIN;

		DROP TABLE IF EXISTS {config["label_frequencies_table"]};

		WITH cte_adm_med AS (
			SELECT 
				visit_key
				, prim_atc AS atc
			FROM {config["adm_med_table"]}
		),
		cte_labels AS (
			SELECT 
				array_to_string(
					array(SELECT DISTINCT unnest(array[a.atc, b.atc])), 
					'_'
				) AS label
			FROM 
				cte_adm_med AS a
				, cte_adm_med AS b
			WHERE a.visit_key = b.visit_key
				AND a.atc <= b.atc -- avoid duplicates in reversed order
		),
		cte_tot AS (
			SELECT count(DISTINCT visit_key)::float as n 
			FROM {config["adm_med_table"]}
		)
		SELECT 
			label AS target_label
			, count(label)/(SELECT n FROM cte_tot) AS freq
		INTO {config["label_frequencies_table"]}
		FROM cte_labels
		GROUP BY label;

		GRANT ALL PRIVILEGES ON {config["label_frequencies_table"]} TO bth_user;
		
		COMMIT;
	"""

	status("Executing query on dbserver")

	with psycopg2.connect(DSN) as conn:
		conn.cursor().execute(q)

	open(snakemake.output[0], "w").write(str(datetime.now()))
	
	status("Done")
