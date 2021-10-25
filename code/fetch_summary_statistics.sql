set search_path to 'nlpandmed';

SET search_path TO nlpandmed;

SELECT count(DISTINCT txt_id) AS n_included_notes_in_admissions
	FROM keras_data
	LEFT JOIN adm using(visit_key)
	LEFT JOIN txt 
		ON txt.pid = adm.pid
		AND datetime BETWEEN adm_datetime AND adm_datetime + '48 hours'::INTERVAL;

WITH cte AS (
	SELECT distinct(pid) FROM keras_data
		LEFT JOIN adm using(visit_key)
)
SELECT count(*) AS n_notes_tot_in_admissions
FROM txt
WHERE pid IN (SELECT pid FROM cte)

WITH cte AS (
	SELECT 
		EXTRACT(YEAR FROM adm_datetime)::int AS adm_year
		, extract(epoch FROM adm_datetime - date_of_birth) / (60 * 60 * 24 * 365.25) AS age 
	FROM adm
	INNER JOIN cpr_210312.person 
		ON person.person_id = adm.pid
	WHERE visit_key IN (SELECT visit_key FROM keras_data)
)
SELECT 
	--adm_year
	percentile_cont(0.5) WITHIN GROUP (ORDER BY age) AS median
	, percentile_cont(0.25) WITHIN GROUP (ORDER BY age) AS p25
	, percentile_cont(0.75) WITHIN GROUP (ORDER BY age) AS p75
FROM cte
--GROUP BY adm_year
--ORDER BY adm_year

SELECT count(*) AS n_preexisting_prescriptions FROM adm_med
WHERE visit_key IN (SELECT visit_key FROM keras_data);

WITH cte AS (
	SELECT
		visit_key
		, count(prim_atc) AS n_drugs
	FROM adm_med
	WHERE visit_key IN (SELECT visit_key FROM keras_data)
	GROUP BY visit_key
)
SELECT 
	sum(CASE WHEN n_drugs >= 5 THEN 1 ELSE 0 END) AS n_polypharmacy_admissions
	, percentile_cont(0.5) WITHIN GROUP (ORDER BY n_drugs asc) AS n_drugs_median
	, percentile_cont(0.25) WITHIN GROUP (ORDER BY n_drugs asc) AS n_drugs_p25
	, percentile_cont(0.75) WITHIN GROUP (ORDER BY n_drugs asc) AS n_drugs_p75
FROM cte;

SELECT * FROM keras_data;

select
	count(distinct keras_data.visit_key) as n_tot
	, avg(case when sex = 'Female' then 1 else 0 end) as prop_female
	, sum(case when sex = 'Female' then 1 else 0 end) as n_female
	, sum(CARDINALITY(tokens)) as n_tokens
	, min(adm_datetime)::date as earliest_admission
	, max(adm_datetime)::date as latest_admissions
from keras_data
left join adm using(visit_key)
left join cpr_210312.person as t_person
	on t_person.person_id = adm.pid;

WITH cte AS (
	SELECT CARDINALITY(tokens) AS n_tokens FROM keras_data
)
SELECT 
	percentile_cont(0.5) WITHIN GROUP (ORDER BY n_tokens asc) AS n_tokens_median
	, percentile_cont(0.25) WITHIN GROUP (ORDER BY n_tokens asc) AS n_tokens_p25
	, percentile_cont(0.75) WITHIN GROUP (ORDER BY n_tokens asc) AS n_tokens_p75
FROM cte;

-- ANNUAL COVERAGES
-- biochem annual coverage
SELECT 
	db_source AS data_source
	, y
	, count(DISTINCT pid) AS n_patients
FROM (SELECT *, extract(YEAR FROM drawn_datetime) AS y FROM biochem_210618.tests) x
GROUP BY y, db_source
--
UNION ALL
--
-- lpr-inpatient annual coverage
SELECT 
	'lpr_inpatient' AS data_source
	, y
	, count(DISTINCT person_id) AS n_patient
FROM (SELECT *, extract(YEAR FROM arrival_date) AS y FROM lpr_210312.adm) x
WHERE visit_type = 0
	AND arrival_date BETWEEN '2006-01-01' AND '2016-07-01'
GROUP BY y
--
UNION ALL
--
-- lpr-outpatient annual coverages
SELECT 
	'lpr_outpatient' AS data_source
	, y
	, count(DISTINCT person_id) AS n_patient
FROM (SELECT *, extract(YEAR FROM arrival_date) AS y FROM lpr_210312.adm) x
WHERE visit_type != 0
	AND arrival_date BETWEEN '2006-01-01' AND '2016-07-01'
GROUP BY y
--
UNION ALL
--
-- medication annual coverage
SELECT 
	"source" AS data_source
	, y
	, count(DISTINCT person_id) AS n_patient
FROM (SELECT *, extract(YEAR FROM adm_date) AS y FROM med_210326.administrations) x
GROUP BY y, "source"


-- then pivot into array (with names ideally)
SELECT 
	p AS percentile
	, percentile_cont(p) WITHIN GROUP (ORDER BY coalesce(CARDINALITY(labels), 0)) AS n_labels
	, percentile_cont(p) WITHIN GROUP (ORDER BY coalesce(CARDINALITY(tokens), 0)) AS n_tokens
FROM nlpandmed.keras_data
	, (SELECT unnest(array[0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95]) AS p) perc
GROUP BY p;
