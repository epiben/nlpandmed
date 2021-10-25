source("code/utils.R")

status("Downloading from database")
conn <- connect()
df <- glue("
	SELECT target_label, auroc, intercept, slope 
	FROM {snakemake@config$eval_table}") %>% 
	sql_fetch(conn) %>% 
	arrange(auroc)
dbDisconnect(conn)

status("Building plot")
p <- ggplot(df, aes(x = intercept, y = slope, colour = auroc)) +
	# geom_vline(xintercept = 0, size = 0.4, linetype = 2) +
	# geom_hline(yintercept = 1, size = 0.4, linetype = 2) +
	geom_jitter(size = 0.2, alpha = 0.75) +
	annotate(geom = "rect", xmin = -0.05, xmax = 0.05, ymin = 0.95, ymax = 1.05, 
			 fill = NA, size = 0.2, colour = "black") +
	scale_colour_gradient(low = "white", high = "blue", limits = c(0.5, 1.0)) +
	theme_minimal() +
	labs(x = "Intercept of calibration line", 
		 y = "Slope of calibration line", 
		 colour = "AUROC")

status("Saving plot to files")
save_plot(p, snakemake@output[[1]])

