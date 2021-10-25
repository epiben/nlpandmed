source("code/utils.R")

conn <- connect()

#' Draws a fingerprint plot of signals
#'
#' This function draws a circular fingerprint plot highlighting prominent
#' signals for a given term (`the_term`).
#'
#' @param the_term string, name of the term whose fingerprint to plot
#' @param score_cutoff numeric, all scores below this value will be greyed out
#' @param top_n int, picks top-N w.r.t. score value and only plots those
#' @param text_size int, arbitrary size for the term text (to be tweaked)
#' 
#' @return ggplot2 object
#' 
draw_fingerprint <- function(the_term, alpha_var = "odds_ratio", top_n = Inf, label_size = 8, term_size = 12, label_offset = NULL, df) {
	colour_lookup <- atc_colours()
	
	padding <- 0.05
	
	plot_df <- df %>% 
		filter(term == the_term) %>% 
		# sample_n(10) %>%
		arrange(atc) %>% 
		mutate(colour_var = str_sub(atc, 1, 1),
			   colour = colour_lookup[colour_var],
			   alpha_var = !!sym(alpha_var)) %>% 
		filter((max(rank(alpha_var)) - rank(alpha_var) + 1) <= top_n) %>% 
		mutate(id = seq_len(n())) %>% 
		rowwise(id, atc, colour, colour_var, alpha_var) 
	
	rainbow_width <- 0.5
	rainbow_df <- plot_df %>% 
		summarise(offsets = 1 + c(0, 0, rainbow_width, rainbow_width),
				  x = offsets * cospi(0.5 - 2 * (id - c(1, 0, 0, 1)) / max(.$id)),
				  y = offsets * sinpi(0.5 - 2 * (id - c(1, 0, 0, 1)) / max(.$id)),
				  .groups = "drop") 
	
	guidebar_width <- 0.1
	guidebar_df <- plot_df %>% 
		summarise(offsets = 1 + rainbow_width + padding + c(0, 0, guidebar_width, guidebar_width),
				  x = offsets * cospi(0.5 - 2 * (id - c(1, 0, 0, 1)) / max(.$id)),
				  y = offsets * sinpi(0.5 - 2 * (id - c(1, 0, 0, 1)) / max(.$id)),
				  .groups = "drop") 
	
	# guideline_df <- plot_df %>% 
	# 	summarise(offsets = c(1.05, 1.05) * rel_circle_width,
	# 			  x = offsets * cospi(0.5 - 2 * (id - 0:1) / max(.$id)),
	# 			  y = offsets * sinpi(0.5 - 2 * (id - 0:1) / max(.$id)),
	# 			  .groups = "drop") 
	#mutate(colour = ifelse(as.numeric(factor(colour_var)) %% 2, "grey70", "grey90"))
	
	label_df <- plot_df %>% 
		group_by(colour_var) %>% 
		summarise(offset = label_offset %||% 1 + 0.9 * rainbow_width, # + 2 * padding + guidebar_width + 2 * padding,
				  x = offset * cospi(0.5 - 2 * mean(c(id[1]-1, id)) / max(.$id)),
				  y = offset * sinpi(0.5 - 2 * mean(c(id[1]-1, id)) / max(.$id)),
				  .groups = "drop")
	
	# Wiggly line
	# centre_plot <- filter(term_vectors, term == the_term) %>% 
	# 	select(-term_group, -term) %>% 
	# 	pivot_longer(everything(), values_to = "ymax") %>% 
	# 	mutate(x = parse_number(name)) %>% 
	# 	ggplot() +
	# 		geom_col(aes(x, ymax)) +
	# 		theme_void() +
	# 		coord_cartesian(ylim = c(-1, 1))
	
	ggplot(mapping = aes(x, y)) +
		geom_polygon(aes(fill = colour, group = id), guidebar_df, show.legend = FALSE) +
		geom_polygon(aes(group = id, alpha = alpha_var), rainbow_df,
					 fill = "grey50", show.legend = FALSE) +
		# geom_polygon(aes(group = id, alpha = alpha_var), filter(rainbow_df, alpha_var < score_cutoff), 
		# 			 fill = "grey50", show.legend = FALSE) +
		# geom_path(aes(colour = colour, group = colour_var), guideline_df) +
		geom_text(aes(label = colour_var), label_df, hjust = "center", vjust = "center",
				  size = label_size/ggplot2::.pt) +
		annotate(geom = "text", label = toupper(the_term), x = 0, y = 0, size = term_size/ggplot2::.pt) +
		scale_alpha_continuous(range = 0:1, limits = c(1, NA)) +
		scale_colour_identity() +
		scale_fill_identity() +
		# annotation_custom(
		# 	ggplotGrob(centre_plot), xmin = -0.8, xmax = 0.8, ymin = -0.4, ymax = 0
		# ) +
		coord_equal() +
		theme_void()
}


term_vectors <- dbGetQuery(conn, "SELECT * FROM nlpandmed.term_vectors;")

label_freqs <- conn %>% 
	dbGetQuery(paste("SELECT * FROM", config$label_frequencies_table)) %>% 
	mutate(odds_empirical = freq / (1 - freq)) %>% 
	tibble()

df <- sql_fetch("SELECT target_label, term, pred FROM @table 
				 WHERE term = main_term
					-- AND domain = '@domain'
					AND target_label !~ '\\_' -- only single-drug labels",
				table = config$predictions_table,
				domain = "mental",
				conn = conn) %>% 
	mutate(odds_prediction = pred / (1 - pred)) %>% 
	inner_join(label_freqs, by = "target_label") %>% 
	mutate(odds_ratio = odds_prediction / odds_empirical,
		   odds_ratio_ntile = ntile(odds_ratio, 100)) %>% 
	rename(atc = target_label) %>% 
	# left_join(lipsum_results, by = "target_label"NN) %>% 
	# mutate(beats_lipsum = odds_ratio >= cutoff_lipsum) %>% 
	tibble()

terms_in_order <- c("amenore", "galaktore", 
					"hypersalivation", "mundtørhed",
					"tremor", "dystoni", "parkinsonisme", "hyperkinesi",
					"asteni", "akatisi", "indre uro",
					"sedation") %>% 
	c(setdiff(unique(df$term), .))

# Use arbitrary term to create the legend
legend_plot <- draw_fingerprint("tremor", alpha_var = "odds_ratio", label_size = 7, label_offset = 0.875, term_size = 0, df = df)
legend_plot$layers[2] <- NULL # remove grey-scale rainbow of signals

plot <- lapply(terms_in_order,
	   draw_fingerprint, alpha_var = "odds_ratio", label_size = 0, term_size = 7, df = df) %>% 
	reduce(`%+%`) +
	legend_plot +
	plot_layout(ncol = 4)

save_plot(plot, "output/figure_3_fingerprints.pdf") # width: 17 cm, height: 25 cm

