source("frontend/utils/data_loader.R")
source("frontend/utils/evaluation.R")

df <- add_derived(load_all_sessions("data/sessions"))
gt <- load_ground_truth("data/ground_truth.csv")
labeled <- label_rows(df, gt, grace_seconds = 2.5)

cm <- confusion_table(labeled)
cat("\nConfusion matrix (rows = actual, columns = predicted):\n")
print(cm)

res <- metrics_table(cm)
cat("\nOverall accuracy:", round(100 * res$accuracy, 1), "%\n\n")
print(res$per_class)

dir.create("analysis/output", recursive = TRUE, showWarnings = FALSE)
write.csv(as.data.frame(cm), "analysis/output/confusion_matrix.csv", row.names = FALSE)
write.csv(res$per_class, "analysis/output/metrics_per_behavior.csv", row.names = FALSE)
ggsave("analysis/output/confusion_matrix.png", plot_confusion(cm),
       width = 8, height = 6, dpi = 150)
cat("\nSaved confusion_matrix.png and metrics_per_behavior.csv to analysis/output\n")