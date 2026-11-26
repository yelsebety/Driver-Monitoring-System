# Inferential statistics for the Driver Monitoring System
#
# Run from the project root:   Rscript scripts/run_statistics.R
#
# Turns each "Expected Analysis" item from the project brief into a tested claim
# and saves the tables to analysis/output/stats_*.csv, a plain-English summary
# to analysis/output/statistics_summary.txt, and two box plots.
#
# Unit of analysis
#   - Frequency / session tests use EPISODES (consecutive rows with the same
#     behavior merged into one event). Rows are logged about once per second, so
#     neighbouring rows are not independent; counting events avoids inflating n.
#   - Confidence and EAR tests use rows, because they describe the detector's
#     output per measurement. This dependence is noted as a limitation.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})
source("frontend/utils/data_loader.R")

# Sessions S01-S04 were recorded before Eating/Drinking detection existed, so
# including them would compare code versions rather than drivers.
EXCLUDE_SESSIONS <- c("S01", "S02", "S03", "S04")
ALPHA <- 0.05
OUT <- "analysis/output"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
set.seed(42)   # for the Monte Carlo p-values

all_df <- add_derived(load_all_sessions("data/sessions"))
df <- all_df %>% filter(!Session_ID %in% EXCLUDE_SESSIONS)
BEHAVIOR_LEVELS_ALL <- c("Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking")
df$Behavior <- factor(df$Behavior, levels = BEHAVIOR_LEVELS_ALL)
episodes <- make_episodes(df) %>%
  mutate(Behavior = factor(Behavior, levels = BEHAVIOR_LEVELS_ALL))
risky_ep <- episodes %>% filter(Behavior != "Normal") %>% droplevels()

# ---------- helpers ----------
summary_lines <- character()
say <- function(...) {
  line <- paste0(...)
  summary_lines <<- c(summary_lines, line)
  cat(line, "\n")
}
fmt_p <- function(p) ifelse(p < 0.001, "< 0.001", sprintf("= %.3f", p))
verdict <- function(p, yes, no) if (p < ALPHA) yes else no
save_csv <- function(x, name) write.csv(x, file.path(OUT, name), row.names = FALSE)

cramers_v <- function(tab) {
  chi <- suppressWarnings(chisq.test(tab, correct = FALSE))$statistic
  k <- min(dim(tab)) - 1
  as.numeric(sqrt(chi / (sum(tab) * k)))
}

# Wilson 95% confidence interval for a proportion
wilson <- function(x, n, conf = 0.95) {
  z <- qnorm(1 - (1 - conf) / 2)
  p <- x / n
  centre <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  c(centre - half, centre + half)
}

# Epsilon-squared effect size for Kruskal-Wallis
epsilon_sq <- function(kw, n) as.numeric(kw$statistic / (n - 1))

say("DRIVER MONITORING SYSTEM - INFERENTIAL STATISTICS")
say("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M"))
say("Sessions analysed: ", paste(sort(unique(df$Session_ID)), collapse = ", "),
    "  (excluded: ", paste(EXCLUDE_SESSIONS, collapse = ", "), ")")
say("Rows: ", nrow(df), "   Episodes: ", nrow(episodes),
    "   Risky episodes: ", nrow(risky_ep), "   Significance level: ", ALPHA)

# =============================================================================
# 1. Frequency and percentage of each risky behavior
#    H0: all five risky behaviors occur equally often (as episodes)
# =============================================================================
say("\n1. FREQUENCY OF RISKY BEHAVIORS")
freq <- risky_ep %>% count(Behavior, name = "Episodes") %>%
  mutate(Percent = 100 * Episodes / sum(Episodes))
ci <- t(sapply(freq$Episodes, wilson, n = sum(freq$Episodes)))
freq$CI_low_pct <- round(100 * ci[, 1], 1)
freq$CI_high_pct <- round(100 * ci[, 2], 1)
freq$Percent <- round(freq$Percent, 1)
time_share <- df %>% group_by(Behavior) %>%
  summarise(Seconds = round(sum(Duration), 1), .groups = "drop") %>%
  mutate(Time_Percent = round(100 * Seconds / sum(Seconds), 1))
freq <- freq %>% left_join(time_share, by = "Behavior")
print(freq)

gof <- chisq.test(freq$Episodes)
say(sprintf("Chi-square goodness-of-fit: X2(%d) = %.2f, p %s",
            gof$parameter, gof$statistic, fmt_p(gof$p.value)))
say(verdict(gof$p.value,
            paste0("-> Risky behaviors are NOT equally frequent. Most common: ",
                   freq$Behavior[which.max(freq$Episodes)], "."),
            "-> No evidence that the risky behaviors differ in frequency."))
save_csv(freq, "stats_1_behavior_frequency.csv")

# =============================================================================
# 2. Comparison of sessions
#    H0: the mix of risky behaviors is the same in every session
# =============================================================================
say("\n2. SESSION x BEHAVIOR (risky episodes)")
tab_sb <- table(risky_ep$Session_ID, risky_ep$Behavior)
print(tab_sb)
# Many expected counts are < 5, so the p-value is estimated by Monte Carlo
indep <- suppressWarnings(chisq.test(tab_sb, simulate.p.value = TRUE, B = 10000))
v <- cramers_v(tab_sb)
say(sprintf("Chi-square test of independence (Monte Carlo, B = 10000): X2 = %.2f, p %s, Cramer's V = %.2f",
            indep$statistic, fmt_p(indep$p.value), v))
say(verdict(indep$p.value,
            "-> The behavior mix differs between sessions.",
            "-> No evidence that sessions differ in their mix of risky behaviors."),
    sprintf(" Effect size is %s.",
            ifelse(v < 0.1, "negligible", ifelse(v < 0.3, "small",
                                                 ifelse(v < 0.5, "medium", "large")))))
save_csv(as.data.frame.matrix(tab_sb) %>% tibble::rownames_to_column("Session_ID"),
         "stats_2_session_behavior_table.csv")

# Risky share per session (time-based), with Wilson CIs on the row share
sess <- df %>% group_by(Session_ID) %>%
  summarise(Rows = n(), Risky_Rows = sum(Risky),
            Minutes = round(max(Seconds) / 60, 2),
            Risky_Episodes = sum(episodes$Session_ID == first(Session_ID) &
                                   episodes$Behavior != "Normal"),
            .groups = "drop") %>%
  mutate(Risky_Percent = round(100 * Risky_Rows / Rows, 1),
         Events_per_Minute = round(Risky_Episodes / Minutes, 2))
print(sess)
save_csv(sess, "stats_2_session_summary.csv")

# =============================================================================
# 3. Number of risky events per session: Poisson rate model
#    H0: every session has the same rate of risky events per minute
# =============================================================================
say("\n3. RATE OF RISKY EVENTS PER SESSION (Poisson GLM)")
pois_full <- glm(Risky_Episodes ~ Session_ID + offset(log(Minutes)),
                 family = poisson, data = sess)
pois_null <- glm(Risky_Episodes ~ 1 + offset(log(Minutes)),
                 family = poisson, data = sess)
lrt <- anova(pois_null, pois_full, test = "LRT")
p_lrt <- lrt$`Pr(>Chi)`[2]
overall_rate <- exp(coef(pois_null)[1])
dispersion <- sum(residuals(pois_null, type = "pearson")^2) / pois_null$df.residual
say(sprintf("Overall rate: %.2f risky events per minute.", overall_rate))
say(sprintf("Likelihood-ratio test (session effect): Chi2(%d) = %.2f, p %s",
            lrt$Df[2], lrt$Deviance[2], fmt_p(p_lrt)))
say(sprintf("Dispersion of the equal-rate model = %.2f (values well above 1 mean extra session-to-session variation).",
            dispersion))
say(verdict(p_lrt, "-> Sessions differ in how often risky events happen.",
            "-> No evidence that the event rate differs between sessions."))

# =============================================================================
# 4. Duration of events by behavior
#    H0: episodes of every risky behavior last equally long
# =============================================================================
say("\n4. EPISODE DURATION BY BEHAVIOR")
dur <- risky_ep %>% group_by(Behavior) %>%
  summarise(Episodes = n(),
            Mean_s = round(mean(Seconds), 2), SD_s = round(sd(Seconds), 2),
            Median_s = round(median(Seconds), 2),
            Q1_s = round(quantile(Seconds, 0.25), 2),
            Q3_s = round(quantile(Seconds, 0.75), 2),
            Max_s = round(max(Seconds), 2), .groups = "drop")
print(dur)
kw_dur <- kruskal.test(Seconds ~ Behavior, data = risky_ep)
say(sprintf("Kruskal-Wallis: H(%d) = %.2f, p %s, epsilon^2 = %.2f",
            kw_dur$parameter, kw_dur$statistic, fmt_p(kw_dur$p.value),
            epsilon_sq(kw_dur, nrow(risky_ep))))
say(verdict(kw_dur$p.value,
            paste0("-> Episode length depends on the behavior. Longest median: ",
                   dur$Behavior[which.max(dur$Median_s)], "."),
            "-> No evidence that episode length differs between behaviors."))
save_csv(dur, "stats_4_episode_duration.csv")

# =============================================================================
# 5. Confidence scores by behavior
#    Step 1: normality check (decides parametric vs non-parametric)
#    Step 2: H0: confidence has the same distribution for every behavior
# =============================================================================
say("\n5. CONFIDENCE SCORE BY BEHAVIOR")
conf <- df %>% filter(!is.na(Confidence)) %>% droplevels()
shap <- conf %>% group_by(Behavior) %>%
  summarise(n = n(),
            Mean = round(mean(Confidence), 3), SD = round(sd(Confidence), 3),
            Median = round(median(Confidence), 3),
            Shapiro_W = if (n() >= 3 && n() <= 5000 && sd(Confidence) > 0)
              round(shapiro.test(Confidence)$statistic, 3) else NA,
            Shapiro_p = if (n() >= 3 && n() <= 5000 && sd(Confidence) > 0)
              signif(shapiro.test(Confidence)$p.value, 3) else NA,
            .groups = "drop")
print(shap)
n_nonnormal <- sum(shap$Shapiro_p < ALPHA, na.rm = TRUE)
say(sprintf("Shapiro-Wilk: %d of %d behaviors are not normally distributed -> use non-parametric tests.",
            n_nonnormal, nrow(shap)))
kw_conf <- kruskal.test(Confidence ~ Behavior, data = conf)
say(sprintf("Kruskal-Wallis: H(%d) = %.2f, p %s, epsilon^2 = %.2f",
            kw_conf$parameter, kw_conf$statistic, fmt_p(kw_conf$p.value),
            epsilon_sq(kw_conf, nrow(conf))))
# Post-hoc: pairwise Wilcoxon rank-sum, Holm-corrected for multiple comparisons
pw <- suppressWarnings(pairwise.wilcox.test(conf$Confidence, conf$Behavior,
                                            p.adjust.method = "holm", exact = FALSE))
pw_df <- as.data.frame(as.table(pw$p.value)) %>%
  filter(!is.na(Freq)) %>%
  rename(Behavior_1 = Var1, Behavior_2 = Var2, p_holm = Freq) %>%
  mutate(p_holm = signif(p_holm, 3), Significant = p_holm < ALPHA)
print(pw_df)
say(verdict(kw_conf$p.value,
            sprintf("-> Confidence differs between behaviors; %d of %d pairs differ after Holm correction.",
                    sum(pw_df$Significant), nrow(pw_df)),
            "-> No evidence that confidence differs between behaviors."))
save_csv(shap, "stats_5_confidence_by_behavior.csv")
save_csv(pw_df, "stats_5_confidence_pairwise.csv")

ggsave(file.path(OUT, "stats_confidence_boxplot.png"),
       ggplot(conf, aes(Behavior, Confidence, fill = Behavior)) +
         geom_boxplot(show.legend = FALSE, outlier.alpha = 0.4) +
         labs(title = "Confidence score by behavior",
              subtitle = sprintf("Kruskal-Wallis p %s", fmt_p(kw_conf$p.value)),
              x = NULL, y = "Confidence") +
         theme_minimal(),
       width = 8, height = 5, dpi = 150)

# =============================================================================
# 6. Time-based pattern
#    H0: the chance that a row is risky does not change with time in the session
# =============================================================================
say("\n6. TIME PATTERN (logistic regression)")
logit <- glm(Risky ~ Minutes + Session_ID, family = binomial, data = df)
co <- summary(logit)$coefficients["Minutes", ]
or <- exp(co["Estimate"])
or_ci <- exp(co["Estimate"] + c(-1, 1) * qnorm(0.975) * co["Std. Error"])
say(sprintf("Risky ~ Minutes + Session: odds ratio per minute = %.2f (95%% CI %.2f-%.2f), p %s",
            or, or_ci[1], or_ci[2], fmt_p(co["Pr(>|z|)"])))
say(verdict(co["Pr(>|z|)"],
            sprintf("-> Risk %s as a session goes on (controlling for session).",
                    ifelse(or > 1, "increases", "decreases")),
            "-> No evidence that risky behavior becomes more or less likely over the session."))

# Simple version for the report: first half vs second half of each session
halves <- df %>% group_by(Session_ID) %>%
  mutate(Half = ifelse(Seconds <= max(Seconds) / 2, "First half", "Second half")) %>%
  ungroup()
tab_half <- table(halves$Half, ifelse(halves$Risky, "Risky", "Normal"))
print(tab_half)
chi_half <- chisq.test(tab_half)
say(sprintf("First vs second half: risky %.1f%% vs %.1f%%, X2(1) = %.2f, p %s",
            100 * tab_half[1, "Risky"] / sum(tab_half[1, ]),
            100 * tab_half[2, "Risky"] / sum(tab_half[2, ]),
            chi_half$statistic, fmt_p(chi_half$p.value)))
say("   Note: the halves test pools all sessions; the logistic model controls for session,",
    " so if they disagree, trust the logistic model (the difference comes from which sessions are long).")
save_csv(data.frame(Term = "Minutes", Odds_Ratio = round(or, 3),
                    CI_low = round(or_ci[1], 3), CI_high = round(or_ci[2], 3),
                    p_value = signif(co["Pr(>|z|)"], 3),
                    First_half_risky_pct = round(100 * tab_half[1, "Risky"] / sum(tab_half[1, ]), 1),
                    Second_half_risky_pct = round(100 * tab_half[2, "Risky"] / sum(tab_half[2, ]), 1),
                    Halves_chi2_p = signif(chi_half$p.value, 3)),
         "stats_6_time_pattern.csv")

# =============================================================================
# 7. Does EAR separate Drowsy from Normal? (validates the EAR threshold)
#    H0: EAR has the same distribution in Drowsy and Normal rows
# =============================================================================
say("\n7. EAR: DROWSY vs NORMAL")
ear <- df %>% filter(Behavior %in% c("Normal", "Drowsy"), !is.na(EAR), EAR > 0) %>%
  droplevels()
ear_sum <- ear %>% group_by(Behavior) %>%
  summarise(n = n(), Mean = round(mean(EAR), 3), SD = round(sd(EAR), 3),
            Median = round(median(EAR), 3),
            Pct_below_0.21 = round(100 * mean(EAR < 0.21), 1), .groups = "drop")
print(ear_sum)
mw <- wilcox.test(EAR ~ Behavior, data = ear, exact = FALSE)
n1 <- sum(ear$Behavior == "Normal"); n2 <- sum(ear$Behavior == "Drowsy")
rank_biserial <- 1 - 2 * mw$statistic / (n1 * n2)   # +1 = Drowsy always lower
welch <- t.test(EAR ~ Behavior, data = ear)
say(sprintf("Mann-Whitney U = %.0f, p %s, rank-biserial r = %.2f",
            mw$statistic, fmt_p(mw$p.value), abs(rank_biserial)))
say(sprintf("Welch t-test (for comparison): t = %.2f, p %s, mean difference = %.3f",
            welch$statistic, fmt_p(welch$p.value), diff(rev(welch$estimate))))
say(verdict(mw$p.value,
            "-> EAR is significantly lower in Drowsy rows: the EAR rule separates the two states.",
            "-> EAR does not clearly separate Drowsy from Normal."))
save_csv(ear_sum, "stats_7_ear_drowsy_vs_normal.csv")

ggsave(file.path(OUT, "stats_ear_boxplot.png"),
       ggplot(ear, aes(Behavior, EAR, fill = Behavior)) +
         geom_boxplot(show.legend = FALSE, outlier.alpha = 0.4) +
         geom_hline(yintercept = 0.21, linetype = "dashed") +
         annotate("text", x = 2.45, y = 0.215, label = "threshold 0.21", vjust = 0, hjust = 1) +
         coord_cartesian(ylim = c(0, 0.6)) +
         labs(title = "Eye Aspect Ratio: Drowsy vs Normal",
              subtitle = sprintf("Mann-Whitney p %s  (y-axis cut at 0.6; %d readings above it hidden)",
                                 fmt_p(mw$p.value), sum(ear$EAR > 0.6)),
              x = NULL, y = "EAR") +
         theme_minimal(),
       width = 7, height = 5, dpi = 150)

# =============================================================================
say("\nLIMITATIONS")
say("- Rows are logged about once per second, so consecutive rows are correlated;",
    " row-level tests (5, 6, 7) treat them as independent and p-values may be optimistic.")
say("- Sessions were recorded by the project team, not a random sample of drivers.")
say("- The classifier output flickers (median episode = 1 s), which splits one real action into",
    " several short episodes and inflates event counts and rates (tests 1-4).")
say("- Normal confidence measures 'distance from triggering a rule', while the other behaviors use",
    " rule strength or YOLO confidence, so confidence values are not on exactly the same scale.")
say("- Labels are the system's predictions, not verified ground truth.")

writeLines(summary_lines, file.path(OUT, "statistics_summary.txt"))
cat("\nSaved tables, plots and statistics_summary.txt to", OUT, "\n")