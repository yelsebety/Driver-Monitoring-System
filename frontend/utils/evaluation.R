library(dplyr)
library(ggplot2)

BEHAVIOR_LEVELS <- c("Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking")

# Ground truth file: one row per period where you KNOW what you were doing.
# Columns: Session_ID, Start_s, End_s, True_Behavior
# (seconds since the session started; anything not listed counts as Normal)
load_ground_truth <- function(path = "data/ground_truth.csv") {
  gt <- read.csv(path, stringsAsFactors = FALSE, strip.white = TRUE)
  needed <- c("Session_ID", "Start_s", "End_s", "True_Behavior")
  missing_cols <- setdiff(needed, names(gt))
  if (length(missing_cols) > 0) {
    stop("ground truth file is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  gt$Start_s <- as.numeric(gt$Start_s)
  gt$End_s <- as.numeric(gt$End_s)
  bad <- setdiff(gt$True_Behavior, BEHAVIOR_LEVELS)
  if (length(bad) > 0) {
    stop("Unknown True_Behavior: ", paste(bad, collapse = ", "),
         ". Allowed: ", paste(BEHAVIOR_LEVELS, collapse = ", "))
  }
  gt
}

# Gives every detection row its true behavior. The system needs about 1.5-2 s of
# an action before it reports it, so the first 'grace_seconds' of each action
# are ignored (marked Ignore = TRUE) instead of being counted as mistakes.
label_rows <- function(df, gt, grace_seconds = 2.5) {
  df <- df %>% filter(Session_ID %in% unique(gt$Session_ID))
  df$True_Behavior <- "Normal"
  df$Ignore <- FALSE
  for (i in seq_len(nrow(gt))) {
    in_session <- df$Session_ID == gt$Session_ID[i]
    inside <- in_session & df$Seconds >= gt$Start_s[i] & df$Seconds < gt$End_s[i]
    df$True_Behavior[inside] <- gt$True_Behavior[i]
    warmup <- inside & df$Seconds < gt$Start_s[i] + grace_seconds
    df$Ignore[warmup] <- TRUE
  }
  df
}

confusion_table <- function(labeled) {
  d <- labeled %>% filter(!Ignore)
  table(True = factor(d$True_Behavior, levels = BEHAVIOR_LEVELS),
        Predicted = factor(d$Behavior, levels = BEHAVIOR_LEVELS))
}

# Overall accuracy + precision / recall / F1 for each behavior
metrics_table <- function(cm) {
  per_class <- data.frame(
    Behavior = rownames(cm),
    Support = as.numeric(rowSums(cm)),
    Precision = ifelse(colSums(cm) > 0, diag(cm) / colSums(cm), NA),
    Recall = ifelse(rowSums(cm) > 0, diag(cm) / rowSums(cm), NA),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  per_class$F1 <- ifelse(
    is.na(per_class$Precision) | is.na(per_class$Recall) |
      (per_class$Precision + per_class$Recall) == 0,
    NA,
    2 * per_class$Precision * per_class$Recall / (per_class$Precision + per_class$Recall)
  )
  per_class$Precision <- round(per_class$Precision, 3)
  per_class$Recall <- round(per_class$Recall, 3)
  per_class$F1 <- round(per_class$F1, 3)
  list(accuracy = sum(diag(cm)) / sum(cm), per_class = per_class)
}

plot_confusion <- function(cm) {
  d <- as.data.frame(cm)   # columns: True, Predicted, Freq
  ggplot(d, aes(x = Predicted, y = True, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq)) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    scale_y_discrete(limits = rev(BEHAVIOR_LEVELS)) +
    labs(title = "Confusion matrix",
         x = "Predicted by the system", y = "Actual behavior", fill = "Count") +
    theme_minimal()
}