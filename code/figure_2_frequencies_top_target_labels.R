source("code/utils.R")
library(colorspace)
library(scales)

conn <- connect()

n_visits_tot <- sql_fetch("SELECT COUNT(*) FROM @", conn, config$keras_table) %>% 
	as.numeric()

dt <- sql_fetch(
	"
	SELECT 
		label
		, count(label) AS n_visits
		, count(label)::float / @n_tot AS prop_visits
	FROM (SELECT unnest(labels) AS label FROM @t) x
	GROUP BY label
		HAVING count(label) >= @min_label_count
	", conn, 
	t = config$keras_table,
	n_tot = n_visits_tot,
	min_label_count = snakemake@params$min_label_count) 

dt[, is_combo := str_detect(label, "_")]
dt[, rank := frank(-prop_visits, ties.method = "dense"), by = "is_combo"]
setorder(dt, is_combo, rank)

# Keep all single drugs and the same number of top drug-combos
plot_dt <- list(dt[(!is_combo)],
				dt[(is_combo)][seq(nrow(dt[(!is_combo), "rank"]))]) %>% 
	rbindlist() %>% 
	separate(label, into = c("drug1", "drug2"), sep = "_", fill = "right", remove = FALSE) %>% 
	mutate(drug2 = coalesce(drug2, drug1),
		   drug_class1 = str_sub(drug1, 1, 1),
		   drug_class2 = str_sub(drug2, 1, 1),
		   across(c(drug_class1, drug_class2), list(colour = ~ atc_colours(atc_classes, NULL)[.])),
		   colour = hex(mixcolor(0.5, hex2RGB(drug_class1_colour), hex2RGB(drug_class2_colour))),
		   drug_class = ifelse(is_combo, sprintf("%s%s", drug_class1, drug_class2), drug_class1),
		   exposure_type = ifelse(is_combo, "Drug pairs", "Single drugs"),
		   exposure_type = factor(exposure_type, c("Single drugs", "Drug pairs"))) %>% 
	bind_rows(mutate(slice(group_by(arrange(., desc(label)), drug_class), 1), label = paste0(label, "zzz"), colour = NA)) %>% 
	group_by(exposure_type) %>% 
	arrange(paste(drug_class1, drug_class2)) %>% 
	mutate(label = fct_inorder(factor(label)),
		   x_axis = as.numeric(label)) # ensure correct order on x axis

label_dt <- plot_dt %>% 
	group_by(exposure_type, drug_class) %>% 
	summarise(colour = colour[1],
			  xmin = min(x_axis),
			  xmax = max(x_axis),
			  xmid = mean(range(x_axis)),
			  y = 0.4) %>% 
	arrange(xmin) %>% 
	mutate(every_n = 3 - 0:(n() - 1) %% 4,
		   alpha = 0.04 * every_n,
		   y = 10^(log10(y) + every_n * log10(1.5))) # ensure linear difference on log scale

p <- ggplot() +
	geom_rect(aes(xmin = xmin - 0.5, xmax = xmax + 0.5 - 1, ymin = 0, ymax = y), label_dt, alpha = 0.1) +
	geom_text(aes(x = xmid, label = drug_class, y = y, colour = colour), label_dt, angle = 0, size = 6.5/ggplot2::.pt, vjust = -0.2) + 
	geom_linerange(aes(x = x_axis, ymin = 0, ymax = prop_visits, colour = colour), filter(plot_dt, !is.na(prop_visits)), na.rm = TRUE) +
	scale_colour_identity() + #guide = "legend", breaks = plot_dt$colour, labels = plot_dt$drug_class) +
	scale_fill_identity() + #guide = "legend", breaks = plot_dt$colour, labels = plot_dt$drug_class) +
	scale_y_continuous(labels = percent_format(1), trans = pseudo_log_trans(sigma = 0.01, base = 10),
					   breaks = c(0, 1, 2, 5, 10, 20, 30, 40)/100) +
	facet_wrap(~ exposure_type, ncol = 1, scales = "free_x") +
	theme_minimal() +
	theme(axis.text.x = element_blank(),
		  axis.title = element_blank(),
		  panel.grid.major.x = element_blank(),
		  panel.grid.minor = element_blank(),
		  panel.grid.major.y = element_line(size = 0.25))

save_plot(p, "output/figure_2_label_frequencies.pdf") # width = 25, height = 13 cm

# Heatmap plot of drug1 vs drug2 (not used)
dt %>% 
	separate(label, into = c("drug1", "drug2"), sep = "_", fill = "right") %>% 
	mutate(drug2 = coalesce(drug2, drug1)) %>% 
	filter(prop_visits > 0.01) %>% 
	ggplot(aes(drug1, drug2, alpha = prop_visits)) +
		geom_tile(fill = "black")
	

# Not used, and I don't remember what it's supposed to do
# label_df <- group_by(df, colour, drug_class) %>% 
# 	summarise(xmin = min(x) - 0.5,
# 			  xmax = max(x) + 0.5,
# 			  label_x = mean(c(min(x), max(x))),
# 			  y = max(prop_visits),
# 			  .groups = "drop") %>% 
# 	mutate(y = max(y) + 0.01,
# 		   line_colour = ifelse(as.numeric(factor(drug_class)) %% 2, "grey60", "black"))
# 
# ggplot() +
# 	geom_rect(aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = y), 
# 			  filter(label_df, as.numeric(factor(drug_class)) %% 2 == 0), fill = "black", alpha = 0.04) +
# 	geom_col(aes(x = x, y = prop_visits, colour = colour), filter(df, n_visits >= 1000)) +
# 	geom_text(aes(x = label_x, y = y, label = drug_class), label_df, vjust = -0.5, size = 9/.pt) +
# 	scale_color_identity() +
# 	scale_y_continuous(labels = scales::percent) +
# 	theme_minimal() +
# 	theme(panel.grid.major.x = element_blank(),
# 		  panel.grid.minor.x = element_blank(),
# 		  axis.title = element_blank(),
# 		  axis.text.x = element_blank())
	
dbDisconnect(conn)