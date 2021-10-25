# UTILITY AND HELPER FUNCTIONS

# Send output to a file for loggin, if file path given
try(sink(snakemake@log[[1]], append = FALSE, type = "output"), silent = TRUE)
try(sink(snakemake@log[[1]], append = FALSE, type = "messages"), silent = TRUE)

# Load packages
packages <- c("glue",
			  "data.table", "plyr", "tidyverse", "purrr",
			  "patchwork",
			  "rjson", 
			  "RPostgreSQL")

for (p in packages) 
	library(p, character.only = TRUE)

# Set "global" variables to simplify code below
config <- fromJSON(file = "config.json")
config$user <- "benkaa"

# config <- snakemake@config
# params <- snakemake@params

atc_classes <- c("A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V")

# === FUNCTIONS ===

#' Warp a vector to break it's contiguous nature
#'
#' Taken the integer argument, the function returns a new vector in which the
#' first half consists of elements with non-even indicies in the input, and the
#' second half those with even indices. 
#'
#' @param x integer, the vector to warp
#'
#' @return warped vector of the same type as the input vector.
#' 
warp_vector <- function(x, i = seq_along(x) %% 2 == 1) {
	c(x[i], x[!i])
}

# ===

#' Create colours for ATC classes
#'
#' Given a list of the ATC classes, this function returns a new vector whose
#' elements are hexadecimal colours codes and names are the corresponding ATC
#' classes.
#'
#' @param atc_classes character, ATC classes (default is the predefined vector
#'   of all classes)
#' @param warp_vector function, if desired a function that warps the input ATC
#'   classes. Set to NULL if no warping is wanted.
#' 
#' @return named vector
#' 
atc_colours <- function(atc_classes = atc_classes, warp_vector = warp_vector) {
	atc_classes <- c("A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V")
	try(atc_classes <- warp_vector(atc_classes), silent = TRUE)
	setNames(scales::hue_pal(c(0, 270))(length(atc_classes)), atc_classes)
}

# ===

#' Save Snakemake plot outputs as PDF, PNG and RDS files
#'
#' Little helper function to produce three equivalent plot files for the same
#' output. Will loo for `units`, `width`and `height` in the `snakemake` S4
#' object and use reasonable defaults for each of these if unspecified.
#'
#' @param p ggplot2 object to save
#' @param fname character, full path ending in .pdf
#'
#' @return Silent.
#'   
save_plot <- function(p, fname) {
	fname <- str_remove_all(fname, ".pdf")
	
	units <- snakemake@params$units %||% "cm"
	width <- snakemake@params$width %||% 20
	height <- snakemake@params$height %||% 15
	
	ggsave(paste0(fname, ".pdf"), p, units = units, width = width, height = height)
	ggsave(paste0(fname, ".png"), p, units = units, width = width, height = height)
	write_rds(p, paste0(fname, ".ggplot"))
}

# ===

#' Connect to dbserver
#'
#' A simply helper function that returns the connections with standard settings
#' for our database setup. The default values of the two arguments are taken
#' from the `config` objects, which in turns is assumed to exist in the
#' snakemake S4 object. If this isn't the case, the function will fail
#' ungraciously.
#'
#' @param dbname string, name of database
#' @param user string, username.
#'
#' @return connection to database in the form of an S4 object that inherits from 
#' `DBIConnection`.
#'
connect <- function(dbname = config$database, user = config$user) {
	dbConnect(dbDriver("PostgreSQL"), host = "dbserver", port = 5432, 
			  dbname = dbname, user = user)
}

# ===

#' Parse a parametrised query
#'
#' When given a query (as a string or file path), this function will replace
#' variables (prepended by `@`) with those given in `...`.
#' 
#' @param query string, either an actual query or the path to a valid query.
#' @param ... replacements.
#' 
#' @return Parsed SQL query.
#' 
#' @example parse_sql("SELECT * FROM @@table", table = "table_name")
#' 
parse_sql <- function(query, ...) {
	try(query <- read_file(query), silent = TRUE)
	params <- unlist(list(...))
	if (!is.null(params)) {
		query <- str_replace_all(query, setNames(params, paste0("@", names(params))))
	}
	query
}

# ===

#' Execute query on or fetch results from dbserver
#'
#' `sql_exec` executes the query on the server without fetching anything, where
#' as `sql_fech` fetches the result of the query as a `data.table`.
#'
#' @param query query or path to a file containing a query. Can be
#'   parameterised, in which case replacements are given in `...`.
#' @param conn connection object, e.g. output of connect()
#' @param ... replacements for parameterised queries.
#'   
sql_exec <- function(query, conn, ...) { 
	DBI::dbExecute(conn, parse_sql(query, ...))
}

sql_fetch <- function(query, conn, ...) { 
	DBI::dbGetQuery(conn, parse_sql(query, ...)) %>% 
		as.data.table()
}
