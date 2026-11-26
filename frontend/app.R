# Driver Monitoring System - Shiny dashboard
#
# How to run (R GUI):
#   setwd("D:/driver-monitoring-system/Driver-Monitoring-System")
#   shiny::runApp("frontend", launch.browser = TRUE)
#
# Tabs
#   Live Monitor     - reads the newest data/sessions/Sxx_events.csv every second,
#                      so it shows the driver's state while `python -m backend.main` runs
#   Overview         - the 6 analysis charts, for the sessions you tick
#   Statistics       - results of scripts/run_statistics.R (+ a button to re-run it)
#   Session Explorer - one session as a timeline and a table
#
# Uses only shiny, dplyr, ggplot2 and DT.

library(shiny)
library(dplyr)
library(ggplot2)
library(DT)

# runApp("frontend") moves R into the frontend folder, so find the project root
ROOT <- if (dir.exists("data/sessions")) getwd() else normalizePath("..")
SESSIONS_DIR <- file.path(ROOT, "data", "sessions")
OUTPUT_DIR <- file.path(ROOT, "analysis", "output")

source(file.path(ROOT, "frontend", "utils", "data_loader.R"))
source(file.path(ROOT, "frontend", "utils", "plots.R"))

BEHAVIORS <- c("Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking")
BEHAVIOR_COLORS <- c(Normal = "#2e7d32", Drowsy = "#c62828", Distracted = "#ef6c00",
                     `Using Phone` = "#6a1b9a", Eating = "#00838f", Drinking = "#1565c0")

# ---------- small helpers ----------
color_for <- function(behavior, default = "#37474f") {
  if (behavior %in% names(BEHAVIOR_COLORS)) BEHAVIOR_COLORS[[behavior]] else default
}
load_sessions_safe <- function() {
  tryCatch(add_derived(load_all_sessions(SESSIONS_DIR)),
           error = function(e) NULL)
}

newest_session_file <- function() {
  files <- list.files(SESSIONS_DIR, pattern = "_events\\.csv$", full.names = TRUE)
  if (length(files) == 0) return(NULL)
  files[which.max(file.info(files)$mtime)]
}

stat_card <- function(title, value, color = "#37474f") {
  div(class = "card",
      div(class = "card-title", title),
      div(class = "card-value", style = paste0("color:", color), value))
}

# ---------- user interface ----------
ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background: #f5f6f8; }
    .card { background: white; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,.12); }
    .card-title { font-size: 13px; color: #777; text-transform: uppercase; }
    .card-value { font-size: 28px; font-weight: bold; }
    .alert-banner { padding: 18px; border-radius: 8px; font-size: 26px; font-weight: bold;
                    text-align: center; color: white; margin: 10px 0 18px 0; }
    .summary-box { background: white; padding: 16px; border-radius: 8px;
                   font-family: monospace; white-space: pre-wrap; font-size: 13px; }
  "))),
  titlePanel("Driver Monitoring System"),

  tabsetPanel(
    # ---- 1. Live monitor ----
    tabPanel("Live Monitor",
      br(),
      p("Start the detector in a terminal (python -m backend.main). This page reads the newest",
        "session file every second."),
      uiOutput("live_banner"),
      uiOutput("live_cards"),
      h4("Recent events"),
      DTOutput("live_table")
    ),

    # ---- 2. Overview charts ----
    tabPanel("Overview",
      br(),
      fluidRow(
        column(3,
          div(class = "card",
            checkboxGroupInput("ov_sessions", "Sessions to include", choices = NULL),
            helpText("S01-S04 were recorded before eating/drinking detection existed."),
            actionButton("ov_reload", "Reload data")
          ),
          uiOutput("ov_counts")
        ),
        column(9,
          selectInput("ov_chart", "Chart",
            choices = c("Frequency of each behavior" = "frequency",
                        "Risky events per session" = "events",
                        "Duration of risky events per session" = "duration",
                        "Confidence distribution" = "confidence",
                        "When risky behaviors happen" = "time",
                        "Session comparison" = "comparison")),
          plotOutput("ov_plot", height = "480px")
        )
      )
    ),

    # ---- 3. Statistics ----
    tabPanel("Statistics",
      br(),
      p("Results of scripts/run_statistics.R. Press the button after recording new sessions."),
      actionButton("st_run", "Re-run statistics", class = "btn-primary"),
      br(), br(),
      fluidRow(
        column(6, imageOutput("st_ear", height = "auto")),
        column(6, imageOutput("st_conf", height = "auto"))
      ),
      br(),
      div(class = "summary-box", textOutput("st_summary"))
    ),

    # ---- 4. Session explorer ----
    tabPanel("Session Explorer",
      br(),
      fluidRow(
        column(3, div(class = "card", selectInput("se_session", "Session", choices = NULL))),
        column(9, uiOutput("se_cards"))
      ),
      plotOutput("se_timeline", height = "300px"),
      DTOutput("se_table")
    )
  )
)

# ---------- server ----------
server <- function(input, output, session) {

  # All session data; reloaded by the button or after re-running statistics
  reload <- reactiveVal(0)
  observeEvent(input$ov_reload, reload(reload() + 1))
  all_data <- reactive({
    reload()
    load_sessions_safe()
  })

  observe({
    df <- all_data()
    req(df)
    ids <- sort(unique(df$Session_ID))
    default <- setdiff(ids, c("S01", "S02", "S03", "S04"))
    if (length(default) == 0) default <- ids
    updateCheckboxGroupInput(session, "ov_sessions", choices = ids, selected = default,
                             inline = TRUE)
    updateSelectInput(session, "se_session", choices = ids, selected = tail(ids, 1))
  })

  # ---------- 1. Live monitor ----------
  live_data <- reactive({
    invalidateLater(1000)
    path <- newest_session_file()
    if (is.null(path)) return(NULL)
    df <- tryCatch(read.csv(path, stringsAsFactors = FALSE), error = function(e) NULL)
    if (is.null(df) || nrow(df) == 0) return(NULL)
    list(df = df, path = path, modified = file.info(path)$mtime)
  })

  output$live_banner <- renderUI({
    ld <- live_data()
    if (is.null(ld)) return(div(class = "alert-banner", style = "background:#78909c",
                                "No session data yet"))
    last <- tail(ld$df, 1)
    age <- as.numeric(difftime(Sys.time(), ld$modified, units = "secs"))
    if (age > 5) {
      return(div(class = "alert-banner", style = "background:#78909c",
                 sprintf("Detector not running (last update %s, %s)",
                         format(ld$modified, "%H:%M:%S"), basename(ld$path))))
    }
    if (last$Alert == "ON") {
      div(class = "alert-banner",
          style = paste0("background:", color_for(last$Behavior, "#c62828")),
          paste("WARNING:", toupper(last$Behavior)))
    } else {
      div(class = "alert-banner", style = "background:#2e7d32", "Driver attentive")
    }
  })

  output$live_cards <- renderUI({
    ld <- live_data()
    req(ld)
    last <- tail(ld$df, 1)
    color <- color_for(last$Behavior)
    fluidRow(
      column(2, stat_card("Session", last$Session_ID)),
      column(2, stat_card("Behavior", last$Behavior, color)),
      column(2, stat_card("EAR", ifelse(is.na(last$EAR), "-", sprintf("%.2f", last$EAR)))),
      column(2, stat_card("Head pose", last$Head_Pose)),
      column(2, stat_card("Confidence", sprintf("%.2f", last$Confidence))),
      column(2, stat_card("Alert", last$Alert,
                          ifelse(last$Alert == "ON", "#c62828", "#2e7d32")))
    )
  })

  output$live_table <- renderDT({
    ld <- live_data()
    req(ld)
    recent <- ld$df %>% filter(Behavior != "Normal") %>% tail(15)
    recent <- recent[rev(seq_len(nrow(recent))), ]
    datatable(recent, rownames = FALSE,
              options = list(dom = "t", pageLength = 15, ordering = FALSE))
  })

  # ---------- 2. Overview ----------
  ov_data <- reactive({
    df <- all_data()
    req(df, input$ov_sessions)
    df %>% filter(Session_ID %in% input$ov_sessions)
  })

  output$ov_counts <- renderUI({
    df <- ov_data()
    div(class = "card",
        div(class = "card-title", "Selected data"),
        p(nrow(df), "rows,", length(unique(df$Session_ID)), "sessions"),
        p(round(100 * mean(df$Risky), 1), "% of rows are risky"))
  })

  output$ov_plot <- renderPlot({
    df <- ov_data()
    episodes <- make_episodes(df)
    switch(input$ov_chart,
      frequency = plot_frequency(df),
      events = plot_events_per_session(episodes),
      duration = plot_duration_per_session(episodes),
      confidence = plot_confidence(df),
      time = plot_time_pattern(df),
      comparison = plot_session_comparison(df))
  })

  # ---------- 3. Statistics ----------
  stats_version <- reactiveVal(0)
  observeEvent(input$st_run, {
    withProgress(message = "Running statistics...", {
      old <- setwd(ROOT)
      on.exit(setwd(old))
      result <- tryCatch({
        source("scripts/run_statistics.R", local = new.env())
        "ok"
      }, error = function(e) conditionMessage(e))
    })
    if (result == "ok") {
      showNotification("Statistics updated.", type = "message")
    } else {
      showNotification(paste("Error:", result), type = "error", duration = NULL)
    }
    stats_version(stats_version() + 1)
    reload(reload() + 1)
  })

  output$st_summary <- renderText({
    stats_version()
    path <- file.path(OUTPUT_DIR, "statistics_summary.txt")
    if (!file.exists(path)) return("No results yet. Press 'Re-run statistics'.")
    paste(readLines(path, warn = FALSE), collapse = "\n")
  })

  show_image <- function(name) {
    renderImage({
      stats_version()
      path <- file.path(OUTPUT_DIR, name)
      if (!file.exists(path)) return(list(src = "", alt = "Run the statistics first"))
      list(src = path, width = "100%", alt = name)
    }, deleteFile = FALSE)
  }
  output$st_ear <- show_image("stats_ear_boxplot.png")
  output$st_conf <- show_image("stats_confidence_boxplot.png")

  # ---------- 4. Session explorer ----------
  se_data <- reactive({
    df <- all_data()
    req(df, input$se_session)
    df %>% filter(Session_ID == input$se_session)
  })

  output$se_cards <- renderUI({
    df <- se_data()
    ep <- make_episodes(df) %>% filter(Behavior != "Normal")
    fluidRow(
      column(3, stat_card("Duration", sprintf("%.1f min", max(df$Minutes)))),
      column(3, stat_card("Rows", nrow(df))),
      column(3, stat_card("Risky events", nrow(ep), "#c62828")),
      column(3, stat_card("Risky time", sprintf("%.0f%%", 100 * mean(df$Risky))))
    )
  })

  output$se_timeline <- renderPlot({
    df <- se_data() %>%
      mutate(Behavior = factor(Behavior, levels = BEHAVIORS),
             End = Seconds + pmax(Duration, 1))
    ggplot(df, aes(xmin = Seconds, xmax = End, y = Behavior, color = Behavior)) +
      geom_linerange(linewidth = 8, show.legend = FALSE) +
      scale_color_manual(values = BEHAVIOR_COLORS, drop = FALSE) +
      scale_y_discrete(drop = FALSE, limits = rev(BEHAVIORS)) +
      labs(title = paste("Timeline of session", input$se_session),
           x = "Seconds since session start", y = NULL) +
      theme_minimal(base_size = 14)
  })

  output$se_table <- renderDT({
    df <- se_data() %>%
      select(Session_ID, Timestamp, Behavior, EAR, Head_Pose, Confidence, Alert)
    datatable(df, rownames = FALSE, filter = "top",
              options = list(pageLength = 15))
  })
}

shinyApp(ui, server)