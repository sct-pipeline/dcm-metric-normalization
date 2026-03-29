#!/usr/bin/env Rscript
#
# URP-CTREE: Find optimal compression ratio cutoff predicting T2w signal change in DCM patients.
#
# Uses conditional inference trees (ctree) from the partykit package.
# Reference: Hothorn & Zeileis (2015). partykit: A Modular Toolkit for Recursive Partitioning in R.
#            Journal of Machine Learning Research, 16, 3905-3909.
#
# Arguments (passed from Python via subprocess):
#   1: input_csv   — CSV with columns: participant_id, MEAN(compression_ratio), Myelopathy
#   2: output_dir  — Directory where figure and summary CSV are saved
#
# Outputs:
#   <output_dir>/ctree_compression_ratio_C3.png   — tree plot
#   <output_dir>/ctree_compression_ratio_C3_summary.csv — terminal node summary

# ---- Auto-install partykit if missing ----
if (!requireNamespace("partykit", quietly = TRUE)) {
  message("Installing partykit from CRAN...")
  install.packages("partykit", repos = "https://cran.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages(library(partykit))

# ---- Parse arguments ----
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_ctree_compression_ratio.R <input_csv> <output_dir>")
}
input_csv  <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Load data ----
df <- read.csv(input_csv, stringsAsFactors = FALSE)

# R replaces special characters in column names with '.'; rename for clarity
colnames(df) <- gsub("MEAN\\.compression_ratio\\.", "compression_ratio", colnames(df))
colnames(df) <- gsub("MEAN\\(compression_ratio\\)", "compression_ratio", colnames(df))

# Drop rows with missing values
df <- df[!is.na(df$compression_ratio) & df$Myelopathy %in% c("yes", "no"), ]

# Convert outcome to ordered factor (no < yes)
df$Myelopathy <- factor(df$Myelopathy, levels = c("no", "yes"), ordered = FALSE)

n_total <- nrow(df)
n_t2wpos <- sum(df$Myelopathy == "yes")
n_t2wneg <- sum(df$Myelopathy == "no")

# ---- Fit ctree ----
# alpha     : Bonferroni-corrected significance threshold for split selection
# maxdepth  : max tree depth (2 = one primary split + one secondary)
# minbucket : minimum n in any terminal node
set.seed(42)
tree <- ctree(
  Myelopathy ~ compression_ratio,
  data    = df,
  control = ctree_control(alpha = 0.05, maxdepth = 2, minbucket = 5)
)

# ---- Extract split information ----
node_ids   <- nodeids(tree)
inner_ids  <- nodeids(tree, terminal = FALSE)
inner_ids  <- inner_ids[inner_ids != 1L | length(inner_ids) == 1L]  # exclude root if it has no split
inner_ids  <- setdiff(inner_ids, nodeids(tree, terminal = TRUE))

# Collect splits from inner nodes
splits_info <- list()
for (nid in inner_ids) {
  nd <- node_party(tree[[nid]])
  sp <- nd$split
  if (!is.null(sp)) {
    cutoff  <- sp$breaks
    p_value <- info_node(nd)$p.value
    splits_info[[length(splits_info) + 1]] <- list(
      node_id = nid,
      cutoff  = cutoff,
      p_value = p_value
    )
  }
}

# ---- Terminal node summary ----
terminal_ids <- nodeids(tree, terminal = TRUE)
summary_rows <- list()
for (tid in terminal_ids) {
  node_data <- df[predict(tree, type = "node") == tid, ]
  n_node    <- nrow(node_data)
  n_pos     <- sum(node_data$Myelopathy == "yes")
  prop_pos  <- if (n_node > 0) round(n_pos / n_node * 100, 1) else NA
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    terminal_node_id = tid,
    n                = n_node,
    n_T2wplus        = n_pos,
    pct_T2wplus      = prop_pos,
    stringsAsFactors = FALSE
  )
}
summary_df <- do.call(rbind, summary_rows)

# Add split cutoff column (NA for nodes without an associated split)
summary_df$split_cutoff <- NA_real_
summary_df$split_p_value <- NA_real_
for (sp in splits_info) {
  # Associate cutoff with child terminal nodes
  children <- nodeids(tree[[sp$node_id]], terminal = TRUE)
  for (ch in children) {
    idx <- summary_df$terminal_node_id == ch
    summary_df$split_cutoff[idx]  <- sp$cutoff
    summary_df$split_p_value[idx] <- sp$p_value
  }
}

# Save summary CSV
csv_path <- file.path(output_dir, "ctree_compression_ratio_C3_summary.csv")
write.csv(summary_df, csv_path, row.names = FALSE)

# ---- Save tree plot ----
fig_path <- file.path(output_dir, "ctree_compression_ratio_C3.png")
png(fig_path, width = 900, height = 600, res = 120)
plot(tree,
     main = paste0("URP-CTREE: Compression Ratio → T2w Myelopathy (C3, n=", n_total, ")"),
     gp   = gpar(fontsize = 11))
dev.off()

# ---- Print formatted summary to stdout (captured by Python) ----
cat("\n")
cat(strrep("=", 72), "\n")
cat("URP-CTREE: Compression Ratio -> T2w Myelopathy (C3 level)\n")
cat(strrep("=", 72), "\n")
cat(sprintf("Package    : partykit %s (ctree)\n", as.character(packageVersion("partykit"))))
cat("Outcome    : Myelopathy (binary factor: no / yes)\n")
cat("Predictor  : compression_ratio  [MEAN(diameter_AP) / MEAN(diameter_RL)]\n")
cat("Parameters : alpha=0.05, maxdepth=2, minbucket=5\n")
cat(sprintf("Sample     : n=%d DCM patients  (T2w-: %d,  T2w+: %d)\n\n",
            n_total, n_t2wneg, n_t2wpos))

if (length(splits_info) == 0) {
  cat("No significant split found (p >= 0.05). The tree has a single terminal node.\n")
  cat("Conclusion: Compression ratio alone does not reach significance as a binary\n")
  cat("            classifier at this sample size and alpha level.\n")
} else {
  cat("Splits (inner nodes):\n")
  for (i in seq_along(splits_info)) {
    sp <- splits_info[[i]]
    cat(sprintf("  Node %d  |  compression_ratio <= %.4f  |  p = %.4f\n",
                sp$node_id, sp$cutoff, sp$p_value))
  }
  cat("\nTerminal nodes:\n")
  for (i in seq_len(nrow(summary_df))) {
    r <- summary_df[i, ]
    split_str <- if (!is.na(r$split_cutoff))
      sprintf("  [split at %.4f, p=%.4f]", r$split_cutoff, r$split_p_value) else ""
    cat(sprintf("  Node %d  |  n=%-4d  |  T2w+: %d/%d (%.1f%%)%s\n",
                r$terminal_node_id, r$n, r$n_T2wplus, r$n,
                r$pct_T2wplus, split_str))
  }
}

cat(strrep("-", 72), "\n")
cat(sprintf("Figure saved : %s\n", fig_path))
cat(sprintf("Summary CSV  : %s\n", csv_path))
cat(strrep("=", 72), "\n\n")
