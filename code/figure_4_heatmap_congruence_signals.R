source("code/utils.R")

conn <- connect()

cosine_similarity <- function(M) {
	"
	M: matrix for whose columns the cosine similarity is to be computed
	"
	t(M) %*% M / (sqrt(colSums(M^2) %*% t(colSums(M^2))))
}

plot_df <- glue("
	SELECT 
		domain
		, main_term
		, term
		, target_label
		, odds_ratio
	FROM nlpandmed.signals
	WHERE target_label IN (
		SELECT DISTINCT target_label
		FROM nlpandmed.signals
		WHERE signal_rank <= 50
			AND odds_ratio > 1
	)") %>% 
	sql_fetch(conn) %>% 
	unite(term_full, c(domain, main_term, term), sep = "_") %>% 
	pivot_wider(id_cols = target_label, names_from = "term_full", values_from = "odds_ratio") %>% 
	select(-target_label) %>% 
	mutate(across(everything(), ~ (. - mean(.)) / sd(.))) %>%  # adj. cosine similarity (http://www10.org/cdrom/papers/519/node14.html)
	as.matrix() %>% 
	cosine_similarity() %>% 
	as.data.frame() %>% 
	rownames_to_column("target_row") %>% 
	pivot_longer(-target_row, names_to = "target_col", values_to = "cosine_similarity") %>% 
	mutate(across(starts_with("target_"), ~ fct_rev(factor(.)))) # aid plotting

p <- ggplot(mapping = aes(target_row, target_col)) +
	geom_tile(aes(fill = cosine_similarity), plot_df) +
	scale_fill_gradient2() +
	theme_minimal() +
	theme(axis.text.x = element_text(angle = 90, vjust = 0, hjust = 0, size = 8),
		  axis.text.y = element_text(size = 8),
		  axis.ticks = element_blank(),
		  axis.title = element_blank(),
		  legend.position = "left") +
	coord_equal() +
	guides(fill = guide_colourbar(title = "Cosine\nsimilarity\n")) +
	scale_y_discrete(position = "right") +
	scale_x_discrete(position = "top")

save_plot(p, snakemake@output[[1]])
 