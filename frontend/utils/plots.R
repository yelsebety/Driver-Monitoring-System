library(dplyr)
library(ggplot2)

# 1. Frequency and percentage of each behavior
plot_frequency <- function(df) {
  freq <- df %>%
    count(Behavior) %>%
    mutate(Percent = round(100 * n / sum(n), 1))
  ggplot(freq, aes(x = reorder(Behavior, -n), y = n, fill = Behavior)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = paste0(n, " (", Percent, "%)")), vjust = -0.4) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Frequency and percentage of each behavior",
         x = "Behavior", y = "Detections") +
    theme_minimal()
}

# 2a. Number of risky events per session
plot_events_per_session <- function(episodes) {
  ev <- episodes %>%
    filter(Behavior != "Normal") %>%
    count(Session_ID, Behavior)
  ggplot(ev, aes(x = Session_ID, y = n, fill = Behavior)) +
    geom_col(position = "dodge") +
    labs(title = "Number of risky events per session",
         x = "Session", y = "Events") +
    theme_minimal()
}

# 2b. Total duration of risky events per session
plot_duration_per_session <- function(episodes) {
  du <- episodes %>%
    filter(Behavior != "Normal") %>%
    group_by(Session_ID, Behavior) %>%
    summarise(Total_Seconds = sum(Seconds), .groups = "drop")
  ggplot(du, aes(x = Session_ID, y = Total_Seconds, fill = Behavior)) +
    geom_col(position = "dodge") +
    labs(title = "Total duration of risky events per session",
         x = "Session", y = "Seconds") +
    theme_minimal()
}

# 3. Confidence score distribution for each behavior
plot_confidence <- function(df) {
  ggplot(df, aes(x = Confidence, fill = Behavior)) +
    geom_histogram(binwidth = 0.05, boundary = 0, show.legend = FALSE) +
    facet_wrap(~Behavior, scales = "free_y") +
    coord_cartesian(xlim = c(0, 1)) +
    labs(title = "Confidence score distribution per behavior",
         x = "Confidence", y = "Count") +
    theme_minimal()
}

# 4. When do risky behaviors happen (minutes since session start)
plot_time_pattern <- function(df) {
  risky <- df %>% filter(Risky)
  ggplot(risky, aes(x = Minutes, fill = Behavior)) +
    geom_histogram(binwidth = 0.5, boundary = 0) +
    facet_wrap(~Session_ID, scales = "free_x") +
    labs(title = "When risky behaviors occur during each session",
         x = "Minutes since session start", y = "Detections") +
    theme_minimal()
}

# 5. Compare sessions: percentage of detections that were risky
plot_session_comparison <- function(df) {
  comp <- df %>%
    group_by(Session_ID) %>%
    summarise(Risky_Percent = round(100 * mean(Risky), 1), .groups = "drop")
  ggplot(comp, aes(x = Session_ID, y = Risky_Percent, fill = Session_ID)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = paste0(Risky_Percent, "%")), vjust = -0.4) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Risky behavior by session",
         x = "Session", y = "% of detections that were risky") +
    theme_minimal()
}