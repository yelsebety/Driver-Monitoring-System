source("frontend/utils/data_loader.R")
source("frontend/utils/plots.R")

df <- add_derived(load_all_sessions("data/sessions"))
episodes <- make_episodes(df)

cat("Rows:", nrow(df), "  Sessions:", length(unique(df$Session_ID)), "\n")
print(table(df$Behavior))

dir.create("analysis/output", recursive = TRUE, showWarnings = FALSE)

charts <- list(
  behavior_frequency = plot_frequency(df),
  events_per_session = plot_events_per_session(episodes),
  duration_per_session = plot_duration_per_session(episodes),
  confidence_distribution = plot_confidence(df),
  time_pattern = plot_time_pattern(df),
  session_comparison = plot_session_comparison(df)
)

for (name in names(charts)) {
  ggsave(file.path("analysis/output", paste0(name, ".png")),
         charts[[name]], width = 8, height = 5, dpi = 150)
}
cat("Saved", length(charts), "charts to analysis/output\n")