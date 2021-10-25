source("code/utils.R")

atc_map <- read_tsv(read_file(snakemake@input$atc_map)) %>% 
	transmute(atc = ATC.code,
			  drug_name = str_to_sentence(ATC.level.name)) %>% 
	group_by(atc) %>% 
	summarise(drug_name = paste(unique(drug_name), collapse = ", ")) %>% 
	with(setNames(drug_name, atc))

danish_interaction_database <- read_tsv("data/danish_interaction_database.tsv") %>% 
	separate(col = "id", into = c("atc1", "atc2"), sep = "_") %>% 
	transmute(atc1, atc2, 
			  did_interaction = sprintf("%s | Clin. sign.: %s | Evidence: %s", recom, cs, evidenceid)) %>% 
	bind_rows(rename(., atc2 = atc1, atc1 = atc2)) 

# There are only 393 ADRs in the file, so that's too little to be credible
# read_tsv(read_file(snakemake@input$ema_summary_of_product_characteristics), col_names = FALSE) %>% 
# 	separate(col = "X1", into = c("to_remove", "drug_name", "adrs"), sep = "%#%", fill = "right") %>% 
# 	select(drug_name, adrs) %>% 
# 	filter(!is.na(drug_name))

read_tsv(snakemake@input$signals) %>% 
	separate(col = "target_label", into = c("atc1", "atc2"), sep = "_", fill = "right") %>% 
	mutate(is_combo = !is.na(atc2),
		   drug_name1 = atc_map[atc1],
		   drug_name2 = atc_map[atc2]) %>% 
	left_join(danish_interaction_database, by = c("atc1", "atc2")) %>% 
	write_tsv(snakemake@output[[1]])
