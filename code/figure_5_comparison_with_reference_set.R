source("code/utils.R")

# To come from snakemake params
the_term <- "kramper"
atc_classification <- "data/00-raw-symlinks/atc_classification.tsv"


atc_map <- read_tsv(read_file(atc_classification)) %>% 
	transmute(atc = ATC.code,
			  drug_name = str_to_sentence(ATC.level.name)) %>% 
	group_by(atc) %>% 
	summarise(drug_name = paste(unique(drug_name), collapse = ", "))

config <- fromJSON(file = "config.json")
config$user <- "benkaa"

conn <- dbConnect(dbDriver("PostgreSQL"), host = "dbserver", port = 5432, 
				  dbname = config$database, user = config$user)

df <- dbGetQuery(conn, sprintf("SELECT * FROM nlpandmed.scores_multimodel WHERE term = '%s';", the_term)) %>% 
	mutate(odds_score = score / (1 - score)) 

atc_counts <- dbGetQuery(conn, "SELECT * FROM nlpandmed.atc_frequencies;") %>% 
	mutate(prop = count / 3.2e6, # found correct value
		   odds_empirical = prop / (1 - prop)) # approximate for now

res <- inner_join(df, atc_counts, by = "atc") %>% 
	inner_join(atc_map, by = "atc") %>% 
	mutate(odds_ratio = odds_score / odds_empirical) %>% 
	group_by(term) %>% 
	mutate(rank = rank(odds_ratio, ties.method = "max"),
		   rank = max(rank) - rank + 1) %>% # make 1 the "best"
	tibble()

View(arrange(res, desc(odds_ratio)))
