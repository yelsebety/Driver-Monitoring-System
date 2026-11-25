library(dplyr)

REQUIRED_COLS <- c("Session_ID", "Timestamp", "Behavior", "EAR",
                   "Head_Pose", "Confidence", "Alert")
RISKY <- c("Drowsy", "Distracted", "Using Phone", "Eating", "Drinking")

# Reads timestamps like "2026-11-25 08:15:32" or just "08:15:32"
parse_time <- function(x) {
  x <- as.character(x)
  formats <- c("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%H:%M:%S")
  out <- as.POSIXct(rep(NA_real_, length(x)), origin = "1970-01-01", tz = "UTC")
  for (f in formats) {
    missing <- is.na(out)
    if (!any(missing)) break
    out[missing] <- as.POSIXct(x[missing], format = f, tz = "UTC")
  }
  out
}

load_session_csv <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE, strip.white = TRUE)
  missing_cols <- setdiff(REQUIRED_COLS, names(df))
  if (length(missing_cols) > 0) {
    stop(path, " is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  df$Time <- parse_time(df$Timestamp)
  if (all(is.na(df$Time))) stop("Could not read the Timestamp column in ", path)
  df$EAR <- as.numeric(df$EAR)
  df$Confidence <- as.numeric(df$Confidence)
  df
}

load_all_sessions <- function(dir = "data/sessions") {
  files <- list.files(dir, pattern = "_events\\.csv$", full.names = TRUE)
  files <- files[file.info(files)$size > 0]   # skip empty files
  if (length(files) == 0) stop("No session CSV files with data found in ", dir)
  bind_rows(lapply(files, load_session_csv))
}

# Adds: Seconds/Minutes since session start, Duration of each row (time until
# the next row), and Risky (TRUE for anything that is not Normal)
add_derived <- function(df) {
  df %>%
    arrange(Session_ID, Time) %>%
    group_by(Session_ID) %>%
    mutate(
      Seconds = as.numeric(difftime(Time, min(Time), units = "secs")),
      Minutes = Seconds / 60,
      Duration = as.numeric(difftime(lead(Time), Time, units = "secs")),
      Duration = ifelse(is.na(Duration), 0, Duration),
      Risky = Behavior %in% RISKY
    ) %>%
    ungroup()
}

# Merges consecutive rows with the same behavior into one event with a duration
make_episodes <- function(df) {
  df %>%
    group_by(Session_ID) %>%
    mutate(Episode = cumsum(Behavior != lag(Behavior, default = first(Behavior))) + 1L) %>%
    group_by(Session_ID, Episode, Behavior) %>%
    summarise(Start = min(Time),
              Seconds = sum(Duration),
              Mean_Confidence = mean(Confidence),
              .groups = "drop")
}