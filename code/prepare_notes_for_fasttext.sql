BEGIN;

SET SEARCH_PATH TO :'schema';

\echo CREATING TEMP VIEW 
CREATE TEMP VIEW output AS (
    SELECT 
    	lower(
    		regexp_replace(
    			regexp_replace(text_t1, '\s+', ' ', 'gi'), 
    		'[´`''%\[\]\%_\-:*\".!?,/(){}]', '', 'gi')
    	) AS note_clean
    FROM epj_210302.txt
);

\echo COPYING TO FILE
\copy (SELECT * FROM output) TO 'data/full_corpus__wo_spec_chars__lowercase.tsv' 
	DELIMITER E'\t' CSV;

COMMIT;
