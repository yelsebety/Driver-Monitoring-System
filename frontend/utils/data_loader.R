library(dplyr)

REQUIRED_COLS <- c(
  "Session_ID", "Timestamp", "Behavior", "EAR",
  "Head_Pose", "Confidence", "Alert"
)

RISKY <- c(
  "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking"
)

# ---------------------------------------------------------
# 1. TIMESTAMP PARSING
# ---------------------------------------------------------

parse_time <- function(x) {
  x <- as.character(x)

  formats <- c(
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%H:%M:%S"
  )

  out <- as.POSIXct(
    rep(NA_real_, length(x)),
    origin = "1970-01-01",
    tz = "UTC"
  )

  for (f in formats) {
    missing <- is.na(out)

    if (!any(missing)) break

    out[missing] <- as.POSIXct(
      x[missing],
      format = f,
      tz = "UTC"
    )
  }

  out
}


# ---------------------------------------------------------
# 2. LOAD ONE SESSION
# ---------------------------------------------------------

load_session_csv <- function(path) {

  df <- read.csv(
    path,
    stringsAsFactors = FALSE,
    strip.white = TRUE
  )

  missing_cols <- setdiff(REQUIRED_COLS, names(df))

  if (length(missing_cols) > 0) {
    stop(
      path,
      " is missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  df$Time <- parse_time(df$Timestamp)

  if (all(is.na(df$Time))) {
    stop("Could not read the Timestamp column in ", path)
  }

  df$EAR <- as.numeric(df$EAR)
  df$Confidence <- as.numeric(df$Confidence)

  df
}


# ---------------------------------------------------------
# 3. FIX 1: REMOVE TERMINAL NO-FACE TAIL
# ---------------------------------------------------------

# Removes a continuous no-face section ONLY when it appears
# at the end of a session.
#
# A row is considered no-face when:
#   - EAR is NA
#   - Head_Pose indicates Away / No Face
#
# Internal no-face periods are preserved.
# Sessions containing no detected face at all are preserved.
#
# min_tail_seconds = 2 means that a terminal no-face section
# must last at least 2 seconds before it is removed.

trim_terminal_no_face <- function(df, min_tail_seconds = 2) {

  if (nrow(df) == 0) {
    return(df)
  }

  if (min_tail_seconds < 0) {
    stop("min_tail_seconds cannot be negative.")
  }

  required <- c("Session_ID", "Time", "EAR", "Head_Pose")

  missing_cols <- setdiff(required, names(df))

  if (length(missing_cols) > 0) {
    stop(
      "Cannot trim no-face tail. Missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  sessions <- df %>%
    arrange(Session_ID, Time) %>%
    group_by(Session_ID) %>%
    group_split(.keep = TRUE)

  cleaned_sessions <- lapply(sessions, function(session) {

    session <- arrange(session, Time)

    n <- nrow(session)

    if (n < 2) {
      return(session)
    }

    # Normalize the head-pose labels.
    head_pose <- toupper(trimws(as.character(session$Head_Pose)))

    # Identify rows where the face is not detected.
    no_face <- is.na(session$EAR) &
      head_pose %in% c("AWAY", "NO FACE", "NO_FACE", "NO-FACE", "NONE")

    # The session does not end with a no-face period.
    if (!no_face[n]) {
      return(session)
    }

    # Find the last row where a face was detected.
    valid_face_rows <- which(!no_face)

    # Do not delete an entire session if no face was detected
    # anywhere in it.
    if (length(valid_face_rows) == 0) {
      return(session)
    }

    last_valid_row <- max(valid_face_rows)

    # Everything after the last valid face detection is the
    # terminal no-face section.
    tail_rows <- seq.int(last_valid_row + 1L, n)

    if (length(tail_rows) == 0) {
      return(session)
    }

    # Estimate the sampling interval using the median timestamp
    # difference. This accounts for the final sample, whose
    # duration cannot be calculated from a following row.
    times <- as.numeric(session$Time)

    intervals <- diff(times)

    intervals <- intervals[
      is.finite(intervals) & intervals > 0
    ]

    sample_interval <- if (length(intervals) > 0) {
      median(intervals)
    } else {
      0
    }

    tail_times <- times[tail_rows]

    tail_times <- tail_times[is.finite(tail_times)]

    if (length(tail_times) == 0) {
      return(session)
    }

    tail_duration <- max(tail_times) -
      min(tail_times) +
      sample_interval

    # Remove the terminal no-face section only if it reaches
    # the configured minimum duration.
    if (tail_duration >= min_tail_seconds) {

      session <- session[
        seq_len(last_valid_row),
        ,
        drop = FALSE
      ]
    }

    session
  })

  bind_rows(cleaned_sessions)
}


# ---------------------------------------------------------
# 4. LOAD ALL SESSIONS
# ---------------------------------------------------------

load_all_sessions <- function(dir = "data/sessions") {

  files <- list.files(
    dir,
    pattern = "_events\\.csv$",
    full.names = TRUE
  )

  # Skip empty session files.
  files <- files[file.info(files)$size > 0]

  if (length(files) == 0) {
    stop("No session CSV files with data found in ", dir)
  }

  all_sessions <- bind_rows(
    lapply(files, load_session_csv)
  )

  # IMPORTANT:
  # Trim terminal no-face periods BEFORE calculating durations
  # or creating episodes.
  all_sessions <- trim_terminal_no_face(
    all_sessions,
    min_tail_seconds = 2
  )

  all_sessions
}


# ---------------------------------------------------------
# 5. ADD DERIVED COLUMNS
# ---------------------------------------------------------

add_derived <- function(df) {

  df %>%
    arrange(Session_ID, Time) %>%
    group_by(Session_ID) %>%
    mutate(
      Seconds = as.numeric(
        difftime(Time, min(Time), units = "secs")
      ),

      Minutes = Seconds / 60,

      Duration = as.numeric(
        difftime(lead(Time), Time, units = "secs")
      ),

      Duration = ifelse(
        is.na(Duration),
        0,
        Duration
      ),

      Risky = Behavior %in% RISKY
    ) %>%
    ungroup()
}


# ---------------------------------------------------------
# 6. FIX 2: CREATE EPISODES WITH SHORT-GAP TOLERANCE
# ---------------------------------------------------------

# Merges two episodes of the same risky behavior when they
# are separated by a short Normal period.
#
# Example:
#
#   Distracted  -> 2 seconds
#   Normal      -> 1 second
#   Distracted  -> 3 seconds
#
# These become ONE Distracted episode.
#
# The Normal gap is included when deciding whether to merge,
# but it is NOT included in the episode's Seconds or
# Mean_Confidence calculations.
#
# gap_tolerance_seconds = 2 means gaps must be strictly
# shorter than 2 seconds.

make_episodes <- function(df, gap_tolerance_seconds = 2) {

  if (gap_tolerance_seconds < 0) {
    stop("gap_tolerance_seconds cannot be negative.")
  }

  if (nrow(df) == 0) {

    return(data.frame(
      Session_ID = df$Session_ID[0],
      Episode = integer(),
      Behavior = character(),
      Start = df$Time[0],
      Seconds = numeric(),
      Mean_Confidence = numeric()
    ))
  }

  required <- c("Session_ID", "Time", "Behavior", "Confidence")

  missing_cols <- setdiff(required, names(df))

  if (length(missing_cols) > 0) {
    stop(
      "Cannot create episodes. Missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  # Process each session independently.
  sessions <- df %>%
    arrange(Session_ID, Time) %>%
    group_by(Session_ID) %>%
    group_split(.keep = TRUE)

  session_episodes <- lapply(sessions, function(session) {

    session <- arrange(session, Time)

    n <- nrow(session)

    if (n == 0) {
      return(NULL)
    }

    # Keep the original behavior labels so that the actual
    # detected behavior duration is not inflated by gaps.
    observed_behavior <- as.character(session$Behavior)

    # Get row durations.
    #
    # If Duration does not exist, calculate it from timestamps.
    if ("Duration" %in% names(session)) {

      row_duration <- as.numeric(session$Duration)

      fallback_duration <- c(
        diff(as.numeric(session$Time)),
        0
      )

      missing_duration <- !is.finite(row_duration)

      row_duration[missing_duration] <-
        fallback_duration[missing_duration]

    } else {

      row_duration <- c(
        diff(as.numeric(session$Time)),
        0
      )
    }

    row_duration[
      !is.finite(row_duration) | row_duration < 0
    ] <- 0


    # -----------------------------------------------------
    # Step A: Identify consecutive behavior runs.
    # -----------------------------------------------------

    initial_runs <- rle(observed_behavior)

    run_values <- initial_runs$values
    run_lengths <- initial_runs$lengths

    run_ends <- cumsum(run_lengths)

    run_starts <- c(
      1L,
      head(run_ends, -1L) + 1L
    )

    # This contains the behavior assigned to each row for
    # episode grouping. Original labels remain unchanged.
    episode_behavior <- observed_behavior


    # -----------------------------------------------------
    # Step B: Bridge short Normal gaps.
    # -----------------------------------------------------

    if (length(run_values) > 2) {

      for (i in seq.int(2L, length(run_values) - 1L)) {

        previous_behavior <- run_values[i - 1L]
        current_behavior <- run_values[i]
        next_behavior <- run_values[i + 1L]

        # Only bridge:
        #   Risky behavior -> Normal -> Same risky behavior
        #
        # Never bridge different risky behaviors.
        if (
          is.na(previous_behavior) ||
          is.na(current_behavior) ||
          is.na(next_behavior)
        ) {
          next
        }

        if (current_behavior != "Normal") {
          next
        }

        if (previous_behavior != next_behavior) {
          next
        }

        if (!(previous_behavior %in% RISKY)) {
          next
        }

        # Get the rows belonging to the Normal gap.
        gap_rows <- seq.int(
          run_starts[i],
          run_ends[i]
        )

        # Calculate the actual duration of the gap.
        gap_duration <- sum(
          row_duration[gap_rows],
          na.rm = TRUE
        )

        # Merge only when the gap is strictly shorter than
        # the configured tolerance.
        if (gap_duration < gap_tolerance_seconds) {

          episode_behavior[gap_rows] <- previous_behavior
        }
      }
    }


    # -----------------------------------------------------
    # Step C: Generate final episode IDs.
    # -----------------------------------------------------

    final_runs <- rle(episode_behavior)

    episode_ids <- rep(
      seq_along(final_runs$lengths),
      final_runs$lengths
    )

    episode_rows <- split(
      seq_len(n),
      episode_ids
    )


    # -----------------------------------------------------
    # Step D: Calculate episode statistics.
    # -----------------------------------------------------

    episode_summaries <- lapply(
      seq_along(episode_rows),
      function(i) {

        rows <- episode_rows[[i]]

        current_behavior <- episode_behavior[rows[1]]

        # Only count rows where the original detected behavior
        # matches the episode behavior.
        #
        # This prevents short Normal gaps from inflating
        # the actual behavior duration.
        matching_rows <- which(
          !is.na(observed_behavior[rows]) &
            observed_behavior[rows] == current_behavior
        )

        actual_behavior_rows <- rows[matching_rows]

        actual_seconds <- sum(
          row_duration[actual_behavior_rows],
          na.rm = TRUE
        )

        confidence_values <- as.numeric(
          session$Confidence[actual_behavior_rows]
        )

        confidence_values <- confidence_values[
          is.finite(confidence_values)
        ]

        mean_confidence <- if (length(confidence_values) > 0) {
          mean(confidence_values)
        } else {
          NA_real_
        }

        data.frame(
          Session_ID = session$Session_ID[1],

          Episode = as.integer(i),

          Behavior = current_behavior,

          Start = min(
            session$Time[rows],
            na.rm = TRUE
          ),

          Seconds = actual_seconds,

          Mean_Confidence = mean_confidence,

          stringsAsFactors = FALSE
        )
      }
    )

    bind_rows(episode_summaries)
  })

  bind_rows(session_episodes)
}