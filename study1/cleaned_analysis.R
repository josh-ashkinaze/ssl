# SSL Study 1 — Analysis
# Converted from cleaned_analysis.ipynb

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — set before running
# ═══════════════════════════════════════════════════════════════════════════

# Which variables to rake on. Options:
#   "union"        – variables meeting EITHER gap ≥ 5 pp OR predicts any SSL
#   "intersection" – variables meeting BOTH  gap ≥ 5 pp AND predicts any SSL
RAKE_MODE <- "union"

# ═══════════════════════════════════════════════════════════════════════════

# ── packages ──────────────────────────────────────────────────────────────
library(tidyverse)
library(survey)
library(ggplot2)
library(patchwork)
library(broom)
library(wCorr)
library(car)
# car/MASS can mask dplyr::select when MASS is lazily loaded during model fitting
select   <- dplyr::select
filter   <- dplyr::filter
mutate   <- dplyr::mutate
rename   <- dplyr::rename
count    <- dplyr::count
map      <- purrr::map
map_chr  <- purrr::map_chr
map_dbl  <- purrr::map_dbl
map_dfr  <- purrr::map_dfr
map_dfc  <- purrr::map_dfc
map_int  <- purrr::map_int
map_lgl  <- purrr::map_lgl
imap     <- purrr::imap
imap_dfr <- purrr::imap_dfr
walk     <- purrr::walk

# ── paths ─────────────────────────────────────────────────────────────────
DATA_FILE  <- "SSL-Final+(Actual+Run)_May+30,+2026_12.59.csv"
QUOTA_FILE <- "yougov_quota.csv"
OUTPUT_DIR <- "analysis_outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# ── aesthetics ────────────────────────────────────────────────────────────
aesthetic_palette_ggplot <- function() {
  c("#00A896","#F6AE2D","#F26419","#F45B69","#2F4858",
    "#86BBD8","#33658A","#020887","#758E4F","#A23B72",
    "#C73E1D","#3B1F2B","#44BBA4","#E94F37","#393E41")
}
PALETTE <- aesthetic_palette_ggplot()

DOMAIN_COLORS <- c(
  personal_psychological = PALETTE[1],   # teal    #00A896
  societal_conventional  = PALETTE[7],   # steel blue #33658A
  moral                  = PALETTE[3]    # orange  #F26419
)
DOMAIN_DISPLAY <- c(
  personal_psychological = "Personal",
  societal_conventional  = "Conventional",
  moral                  = "Moral"
)
DOMAIN_ORDER <- names(DOMAIN_DISPLAY)

BUCKET_ORDER   <- c("practical","productive","social","entertain")
BUCKET_DISPLAY <- c(practical="Practical", productive="Productivity",
                    social="Social",       entertain="Entertainment")
BUCKET_COLORS  <- setNames(PALETTE[c(6,7,9,13)], BUCKET_ORDER)

theme_aesthetic_ggplot <- function(hex_color_list = NULL,
                                   bold_title = FALSE,
                                   show_grid = TRUE,
                                   font_scale = 1,
                                   font_hierarchy_ratio = 1.25,
                                   smallest_font_size = 10,
                                   base_family = "Helvetica") {
  if (is.null(hex_color_list)) hex_color_list <- aesthetic_palette_ggplot()
  small  <- smallest_font_size
  medium <- small * font_hierarchy_ratio
  large  <- medium * font_hierarchy_ratio
  text_color <- "#2C3531"
  grid_color <- "#e8e8e8"

  ggplot2::theme_minimal(base_family = base_family, base_size = small * font_scale) +
    ggplot2::theme(
      text            = ggplot2::element_text(family = base_family, colour = text_color),
      axis.title      = ggplot2::element_text(size = medium * font_scale, face = "bold",
                          colour = text_color, margin = ggplot2::margin(t=4*font_scale, r=4*font_scale)),
      axis.title.x    = ggplot2::element_text(margin = ggplot2::margin(t=8*font_scale)),
      axis.title.y    = ggplot2::element_text(margin = ggplot2::margin(r=8*font_scale)),
      axis.text       = ggplot2::element_text(size = small * font_scale, colour = text_color),
      legend.title    = ggplot2::element_text(size = small * font_scale, face = "bold",
                          colour = text_color, margin = ggplot2::margin(b=4*font_scale)),
      legend.text     = ggplot2::element_text(size = small * font_scale, colour = text_color),
      plot.title      = ggplot2::element_text(size = large * font_scale, face = "bold",
                          hjust = 0, colour = text_color, lineheight = 1.05,
                          margin = ggplot2::margin(t=10, r=5, b=10, l=5)),
      plot.subtitle   = ggplot2::element_text(size = medium * font_scale, hjust = 0,
                          colour = text_color, lineheight = 1.1,
                          margin = ggplot2::margin(b=8*font_scale)),
      plot.title.position   = "plot",
      plot.caption.position = "plot",
      panel.background  = ggplot2::element_rect(fill = "white", colour = NA),
      plot.background   = ggplot2::element_rect(fill = "white", colour = NA),
      axis.line         = ggplot2::element_blank(),
      axis.ticks        = ggplot2::element_blank(),
      panel.border      = ggplot2::element_blank(),
      panel.grid.major  = if (show_grid)
        ggplot2::element_line(colour = grid_color, linewidth = 0.5, linetype = "longdash")
        else ggplot2::element_blank(),
      panel.grid.minor  = ggplot2::element_blank(),
      panel.spacing.x   = ggplot2::unit(2, "lines"),
      legend.background = ggplot2::element_rect(fill = "white", colour = NA),
      legend.key        = ggplot2::element_rect(fill = "white", colour = "white"),
      legend.key.width  = ggplot2::unit(1.5, "lines"),
      legend.spacing    = ggplot2::unit(6, "pt"),
      legend.box.spacing = ggplot2::unit(8, "pt"),
      legend.position   = "right",
      plot.margin = ggplot2::margin(t=12*font_scale, r=14*font_scale,
                                    b=12*font_scale, l=12*font_scale)
    )
}

# ═══════════════════════════════════════════════════════════════════════════
# 1  Pre-process
# ═══════════════════════════════════════════════════════════════════════════

# Qualtrics CSVs: row 1 = col names, rows 2-3 = label rows → skip them
col_names <- names(read_csv(DATA_FILE, n_max = 0, show_col_types = FALSE))
df_raw    <- read_csv(DATA_FILE, skip = 3, col_names = col_names, show_col_types = FALSE)
cat("Raw:", nrow(df_raw), "rows x", ncol(df_raw), "cols\n")

# ── recoding helpers ───────────────────────────────────────────────────────

extract_leading_number <- function(x) {
  x   <- as.character(x); x[x == ""] <- NA_character_
  m   <- regexpr("^-?\\d+(\\.\\d+)?", x, perl = TRUE)
  res <- rep(NA_real_, length(x))
  hit <- !is.na(m) & m > 0          # guard against NA inputs
  res[hit] <- as.numeric(regmatches(x, m)[hit])
  res
}

yes_no_ssl <- function(x) {
  x <- trimws(as.character(x))
  case_when(is.na(x) | x == "NA" ~ NA_real_,
            x == "No - never"    ~ 0, TRUE ~ 1)
}

frequency_code <- function(x) {
  x <- trimws(as.character(x))
  case_when(
    x == "No - never"               ~ 0,
    x == "Yes - once or twice"      ~ 1,
    x == "Yes - 3-10 times"         ~ 2,
    x == "Yes - 11-20 times"        ~ 3,
    x == "Yes - more than 20 times" ~ 4,
    TRUE ~ NA_real_
  )
}

recode_sias <- function(x) {
  m <- c("Not at all characteristic or true of me"=0,
         "Slightly characteristic or true of me"  =1,
         "Moderately characteristic or true of me"=2,
         "Very characteristic or true of me"      =3,
         "Extremely characteristic or true of me" =4)
  unname(m[trimws(as.character(x))])
}

recode_lsns <- function(x) {
  m <- c("None"=0,"0"=0,"1"=1,"One"=1,"2"=2,"Two"=2,
         "3 or 4"=3,"3 or 4 people"=3,
         "5 to 8"=4,"5 to 8 people"=4,
         "9 or more"=5,"9 or more people"=5)
  unname(m[trimws(as.character(x))])
}

recode_tipi <- function(x) {
  m <- c("Disagree strongly"=1,"Disagree moderately"=2,"Disagree a little"=3,
         "Neither agree nor disagree"=4,
         "Agree a little"=5,"Agree moderately"=6,"Agree strongly"=7)
  unname(m[trimws(as.character(x))])
}

rev_score <- function(x, lo, hi) lo + hi - x

recode_air <- function(x) {
  m <- c("Never"=0,"Less than a few times a month"=1,"A few times a month"=2,
         "Once a week"=3,"A few times a week"=4,"Once a day"=5,"Several times a day"=6)
  unname(m[trimws(as.character(x))])
}

recode_ais <- function(x) {
  m <- c("No, never"=0,"Yes, once or twice"=1,"Yes, occasionally"=2,"Yes, frequently"=3)
  unname(m[trimws(as.character(x))])
}

standardize <- function(x) {
  x  <- as.numeric(x)
  sd <- sd(x, na.rm = TRUE)
  if (is.na(sd) || sd == 0) return(rep(NA_real_, length(x)))
  (x - mean(x, na.rm = TRUE)) / sd
}

mad_sd <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_real_)
  1.4826 * median(abs(x - median(x)))
}

# ── score psychometrics ────────────────────────────────────────────────────

score_aias4 <- function(df) {
  items <- c("AIAS-4_1","AIAS-4_2","AIAS-4_3","AIAS-4_4")
  nums  <- map_dfc(items, ~ tibble(!!paste0(.x,"_num") := extract_leading_number(df[[.x]])))
  df    <- bind_cols(df, nums)
  df$aias4_score <- rowMeans(df[, paste0(items,"_num")], na.rm = FALSE)
  df
}

score_anthrotech <- function(df) {
  items <- paste0("AnthroTech_", 1:8)
  nums  <- map_dfc(items, ~ tibble(!!paste0(.x,"_num") := extract_leading_number(df[[.x]])))
  df    <- bind_cols(df, nums)
  df$anthrotech_score <- rowMeans(df[, paste0(items,"_num")], na.rm = FALSE)
  df
}

score_sias4 <- function(df) {
  items <- c("SIAS-4_6","SIAS-4_3","SIAS-4_8","SIAS-4_16","SIAS-4_18","SIAS-4_19")
  nums  <- map_dfc(items, ~ tibble(!!paste0(.x,"_num") := recode_sias(df[[.x]])))
  df    <- bind_cols(df, nums)
  df$sias4_score <- rowSums(df[, paste0(items,"_num")], na.rm = FALSE)
  df
}

score_lsns6 <- function(df) {
  items <- c("LSNS-6-Family_1","LSNS-6-Family_2","LSNS-6-Family_3",
             "LSNS-6-Friends_1","LSNS-6-Friends_2","LSNS-6-Friends_3")
  nums  <- map_dfc(items, ~ tibble(!!paste0(.x,"_num") := recode_lsns(df[[.x]])))
  df    <- bind_cols(df, nums)
  # Qualtrics stores "None" (0 contacts) as a blank cell — NA here means 0,
  # not missing data. Replace before summing so every respondent gets a score.
  num_cols <- paste0(items, "_num")
  df[num_cols] <- lapply(df[num_cols], function(x) replace(x, is.na(x), 0))
  df$lsns6_score <- rowSums(df[, num_cols], na.rm = FALSE)
  df
}

score_tipi <- function(df) {
  for (i in 1:10) df[[paste0("TIPI-",i,"_num")]] <- recode_tipi(df[[paste0("TIPI-",i)]])
  df$tipi_extraversion       <- (df[["TIPI-1_num"]]  + rev_score(df[["TIPI-6_num"]],  1,7)) / 2
  df$tipi_agreeableness      <- (df[["TIPI-7_num"]]  + rev_score(df[["TIPI-2_num"]],  1,7)) / 2
  df$tipi_conscientiousness  <- (df[["TIPI-3_num"]]  + rev_score(df[["TIPI-8_num"]],  1,7)) / 2
  df$tipi_emotional_stability<- (df[["TIPI-9_num"]]  + rev_score(df[["TIPI-4_num"]],  1,7)) / 2
  df$tipi_openness           <- (df[["TIPI-5_num"]]  + rev_score(df[["TIPI-10_num"]], 1,7)) / 2
  df
}

# ── build main dataframe ───────────────────────────────────────────────────
df <- df_raw |>
  mutate(
    duration_seconds      = as.numeric(`Duration (in seconds)`),
    finished_bool         = tolower(Finished) == "true",
    progress_num          = as.numeric(Progress),
    committed             = commit == "Yes, I will",
    eligible_chatbot_user = use_chatbots == "Yes",

    used_moral_ssl        = yes_no_ssl(domain_e_freq),
    used_personal_ssl     = yes_no_ssl(domain_p_freq),
    used_conventional_ssl = yes_no_ssl(domain_c_freq),

    moral_ssl_freq_code        = frequency_code(domain_e_freq),
    personal_ssl_freq_code     = frequency_code(domain_p_freq),
    conventional_ssl_freq_code = frequency_code(domain_c_freq),

    air_friend_num   = recode_air(air_friend),
    air_rship_num    = recode_air(air_rship),
    ais_chosen_num   = recode_ais(ais_chosen)   * (6/3),
    ais_shared_num   = recode_ais(ais_shared)   * (6/3),
    ais_embarass_num = recode_ais(ais_embarass)  * (6/3),
    ais_grief_num    = recode_ais(ais_grief)     * (6/3),

    use_freq_code = case_when(
      use_freq == "Rarely"    ~ 1, use_freq == "Sometimes" ~ 2,
      use_freq == "Frequently"~ 3, use_freq == "Every day" ~ 4,
      TRUE ~ NA_real_),
    age_num  = as.numeric(age),
    ideo_num = case_when(
      ideo5 == "Extremely liberal"           ~ 1, ideo5 == "Liberal"              ~ 2,
      ideo5 == "Slightly liberal"            ~ 3,
      ideo5 == "Moderate; middle of the road"~ 4,
      ideo5 == "Slightly conservative"       ~ 5, ideo5 == "Conservative"         ~ 6,
      ideo5 == "Extremely conservative"      ~ 7, TRUE ~ NA_real_),
    income_ord = case_when(
      income == "Less than $25,000" ~ 1, income == "$25,000-$49,999"   ~ 2,
      income == "$50,000-$74,999"   ~ 3, income == "$75,000-$99,999"   ~ 4,
      income == "$100,000-$149,999" ~ 5, income == "$150,000 or more"  ~ 6,
      TRUE ~ NA_real_),
    college_degree = as.numeric(education %in% c(
      "Bachelor's degree",
      "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS etc.)")),
    gender_male    = as.numeric(gender == "Male"),
    gender_other   = as.numeric(!gender %in% c("Male","Female")),
    race_white     = as.numeric(race == "White or Caucasian"),
    race_black     = as.numeric(race == "Black or African American"),
    pid_republican = as.numeric(pid3 == "Republican"),
    pid_democrat   = as.numeric(pid3 == "Democrat"),
    employed       = as.numeric(grepl("Employed|Self-employed", employment)),
    retired        = as.numeric(employment == "Retired")
  ) |>
  mutate(
    n_ssl_domains = rowSums(cbind(used_personal_ssl, used_conventional_ssl, used_moral_ssl),
                            na.rm = FALSE),
    used_any_ssl  = if_else(
      is.na(used_personal_ssl) & is.na(used_conventional_ssl) & is.na(used_moral_ssl),
      NA_real_,
      as.numeric(used_personal_ssl == 1 | used_conventional_ssl == 1 | used_moral_ssl == 1)
    )
  )

df <- score_aias4(df)
df <- score_anthrotech(df)
df <- score_sias4(df)
df <- score_lsns6(df)
df <- score_tipi(df)

df <- df |>
  mutate(ai_social_use_index = rowMeans(
    cbind(air_friend_num, air_rship_num,
          ais_chosen_num, ais_shared_num, ais_embarass_num, ais_grief_num),
    na.rm = TRUE))

# Speed exclusion: duration < median - 2*mad_sd within n_ssl_domains group
df <- df |>
  group_by(n_ssl_domains) |>
  mutate(
    duration_group_median = median(duration_seconds, na.rm = TRUE),
    duration_group_mad_sd = mad_sd(duration_seconds),
    went_fast_threshold   = duration_group_median - 2 * duration_group_mad_sd,
    went_fast             = duration_seconds < went_fast_threshold
  ) |>
  ungroup()

base_mask     <- df$finished_bool & !is.na(df$progress_num) & df$progress_num >= 100 &
                 df$committed & df$eligible_chatbot_user & !is.na(df$eligible_chatbot_user)
analysis_mask <- base_mask & !(df$went_fast %in% TRUE)
df_analysis   <- df[analysis_mask, ]

tibble(sample = c("raw","base_eligible","analysis"),
       n      = c(nrow(df), sum(base_mask,na.rm=TRUE), nrow(df_analysis))) |> print()

# Standardize
SCALE_COLS <- c("aias4_score","anthrotech_score","sias4_score","lsns6_score",
                "tipi_extraversion","tipi_agreeableness","tipi_conscientiousness",
                "tipi_emotional_stability","tipi_openness","ai_social_use_index",
                "use_freq_code","age_num","ideo_num","income_ord")
for (col in SCALE_COLS) df_analysis[[paste0(col,"_z")]] <- standardize(df_analysis[[col]])

write_csv(df,          file.path(OUTPUT_DIR, "study1_qualtrics_scored_with_flags.csv"))
write_csv(df_analysis, file.path(OUTPUT_DIR, "study1_analysis_sample.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# 2  Weighting
# ═══════════════════════════════════════════════════════════════════════════

quota_df <- read_csv(QUOTA_FILE, show_col_types = FALSE)

# Collapsed quota variables for all candidate raking variables
df_analysis <- df_analysis |>
  mutate(
    quota_llm_freq2 = case_when(
      use_freq %in% c("Every day","Frequently") ~ "High",
      use_freq %in% c("Sometimes","Rarely")     ~ "Low",
      TRUE ~ "Other"),
    quota_age2 = case_when(
      as.numeric(age) >= 18 & as.numeric(age) <= 44 ~ "18-44",
      as.numeric(age) >= 45                          ~ "45+",
      TRUE ~ "Other"),
    quota_income = case_when(
      income %in% c("Less than $25,000","$25,000-$49,999")  ~ "<$50K",
      income %in% c("$50,000-$74,999","$75,000-$99,999")    ~ "$50-100K",
      income %in% c("$100,000-$149,999","$150,000 or more") ~ "$100K+",
      TRUE ~ "Other"),
    quota_gender = case_when(
      gender == "Male"   ~ "Male",
      gender == "Female" ~ "Female",
      TRUE ~ "Other"),
    quota_pid3 = case_when(
      pid3 == "Democrat"    ~ "Democrat",
      pid3 == "Republican"  ~ "Republican",
      pid3 == "Independent" ~ "Independent",
      TRUE ~ "Other")
  )

# Pre-compute population targets for every candidate variable
llm_targets  <- c(High = 0.33, Low = 0.67)

age_yg       <- quota_df |> filter(category == "Age") |>
  mutate(p = yes / sum(yes)) |> select(group, p) |> deframe()
age2_targets <- c("18-44" = age_yg[["18-29"]] + age_yg[["30-44"]],
                  "45+"   = age_yg[["45-64"]] + age_yg[["65+"]])
age2_targets <- age2_targets / sum(age2_targets)

inc_yg       <- quota_df |> filter(category == "Income") |>
  mutate(p = yes / sum(yes)) |> select(group, p) |> deframe()

gender_yg    <- quota_df |> filter(category == "Gender") |>
  mutate(p = yes / sum(yes)) |> select(group, p) |> deframe()

pid3_yg      <- quota_df |> filter(category == "Party ID") |>
  mutate(p = yes / sum(yes)) |> select(group, p) |> deframe()

# ── Raking variable selection ─────────────────────────────────────────────
# Candidate variables evaluated on two criteria:
#   (A) gap  – max absolute gap vs. YouGov LLM-user proportions ≥ 5 pp
#   (B) pred – variable significantly predicts any-SSL (logistic LRT p < .05)
# RAKE_MODE (set at top of script) determines which variables are included:
#   "union"        → gap_crit OR ssl_crit
#   "intersection" → gap_crit AND ssl_crit

yg_pct <- quota_df |>
  group_by(category) |>
  mutate(yg_pct = yes / sum(yes) * 100) |>
  ungroup() |>
  select(category, group, yg_pct)

df_ws <- df_analysis |>
  mutate(
    ws_gender  = case_when(gender == "Male"   ~ "Male",
                           gender == "Female" ~ "Female"),
    ws_age     = case_when(age_num >= 18 & age_num <= 29 ~ "18-29",
                           age_num >= 30 & age_num <= 44 ~ "30-44",
                           age_num >= 45 & age_num <= 64 ~ "45-64",
                           age_num >= 65                 ~ "65+"),
    ws_pid3    = case_when(pid3 == "Democrat"    ~ "Democrat",
                           pid3 == "Republican"  ~ "Republican",
                           pid3 == "Independent" ~ "Independent"),
    ws_income  = quota_income,
    ws_llmfreq = quota_llm_freq2
  )

gap_for <- function(col, yg_cat, manual_targets = NULL) {
  sp <- df_ws |> filter(!is.na(.data[[col]])) |>
    count(level = .data[[col]]) |>
    mutate(sample_pct = n / sum(n) * 100)
  if (!is.null(yg_cat)) {
    yg <- yg_pct |> filter(category == yg_cat) |> select(level = group, yg_pct) |>
      filter(level %in% sp$level) |>
      mutate(yg_pct = yg_pct / sum(yg_pct) * 100)
    sp <- sp |> left_join(yg, by = "level")
  } else {
    sp <- sp |> mutate(yg_pct = manual_targets[level])
  }
  max(abs(sp$sample_pct - sp$yg_pct), na.rm = TRUE)
}

ssl_p_for <- function(col) {
  d   <- df_ws |> filter(!is.na(.data[[col]]), !is.na(used_any_ssl)) |>
    mutate(x = factor(.data[[col]]))
  fit <- glm(used_any_ssl ~ x, data = d, family = binomial)
  drop1(fit, test = "Chisq")[["Pr(>Chi)"]][2]
}

# Each entry: label, quota column name, ws column name, YouGov category / manual targets
all_rake_specs <- list(
  list(label="LLM Freq.", quota_col="quota_llm_freq2", ws_col="ws_llmfreq",
       yg_cat=NULL,        targets=llm_targets,  manual=c(High=33, Low=67)),
  list(label="Age",        quota_col="quota_age2",      ws_col="ws_age",
       yg_cat="Age",       targets=age2_targets, manual=NULL),
  list(label="Income",     quota_col="quota_income",    ws_col="ws_income",
       yg_cat="Income",    targets=inc_yg,       manual=NULL),
  list(label="Gender",     quota_col="quota_gender",    ws_col="ws_gender",
       yg_cat="Gender",    targets=gender_yg,    manual=NULL),
  list(label="Party ID",   quota_col="quota_pid3",      ws_col="ws_pid3",
       yg_cat="Party ID",  targets=pid3_yg,      manual=NULL)
)

wt_sel_tbl <- map_dfr(all_rake_specs, function(s) {
  gap <- gap_for(s$ws_col, s$yg_cat, s$manual)
  p   <- ssl_p_for(s$ws_col)
  tibble(variable   = s$label,
         max_gap_pp = round(gap, 1),
         ssl_lrt_p  = round(p, 4),
         gap_crit   = gap >= 5,
         ssl_crit   = p   < .05)
})

wt_sel_tbl <- wt_sel_tbl |>
  mutate(rake_selected = case_when(
    RAKE_MODE == "union"        ~ gap_crit | ssl_crit,
    RAKE_MODE == "intersection" ~ gap_crit & ssl_crit,
    TRUE ~ FALSE
  ))

cat(sprintf("\n── Raking variable selection  [RAKE_MODE = \"%s\"] ──────────────────────\n",
            RAKE_MODE))
cat("(A) gap_crit: max gap vs. YouGov LLM-user % ≥ 5 pp\n")
cat("(B) ssl_crit: LRT p < .05 for any-SSL prediction\n\n")
print(wt_sel_tbl)

selected_specs  <- all_rake_specs[wt_sel_tbl$rake_selected]
selected_labels <- map_chr(selected_specs, "label")
cat(sprintf("\nRaking on (%s): %s\n\n",
            RAKE_MODE, paste(selected_labels, collapse = ", ")))

# ── Build and run rake dynamically ────────────────────────────────────────
make_pop_df <- function(var, props) {
  data.frame(setNames(list(names(props)), var),
             Freq = as.numeric(props) * nrow(df_analysis))
}

# Save pre-filter snapshot for the sensitivity analysis loop below
df_analysis_base <- df_analysis

# Drop respondents not represented in any selected quota cell
for (s in selected_specs) {
  df_analysis <- dplyr::filter(df_analysis, .data[[s$quota_col]] != "Other")
}

design0 <- survey::svydesign(ids = ~1, weights = ~1, data = df_analysis)
design_raked <- survey::rake(
  design             = design0,
  sample.margins     = purrr::map(selected_specs, ~ as.formula(paste0("~", .x$quota_col))),
  population.margins = purrr::map(selected_specs, ~ make_pop_df(.x$quota_col, .x$targets)),
  control = list(maxit = 500, epsilon = 1e-8, verbose = FALSE)
)

# Cap at 5, then trim at p95
cap_weights <- function(w, cap = 5) {
  n <- length(w)
  repeat {
    over <- w > cap
    if (!any(over)) break
    w[over] <- cap
    w[!over] <- w[!over] * (n - sum(over) * cap) / sum(w[!over])
  }
  w * n / sum(w)
}

trim_weights <- function(w, pct = 95) {
  w <- pmin(w, quantile(w, pct/100))
  w * length(w) / sum(w)
}

df_analysis$weight_raw     <- weights(design_raked)
df_analysis$weight_cap5    <- cap_weights(df_analysis$weight_raw)
df_analysis$weight_trimmed <- trim_weights(df_analysis$weight_cap5)
WGT <- "weight_trimmed"

cat("Weight summary:\n")
w <- df_analysis[[WGT]]
tibble(n = nrow(df_analysis), mean = mean(w), min = min(w), max = max(w),
       cv = sd(w)/mean(w), kish_n = sum(w)^2/sum(w^2)) |> print()

# Final survey design
svy <- svydesign(ids = ~1, weights = as.formula(paste0("~",WGT)), data = df_analysis)

# Balance check — built dynamically from selected raking variables
balance <- map_dfr(selected_specs, function(s) {
  qcol    <- s$quota_col
  targets <- s$targets
  tibble(var   = s$label,
         level = names(targets),
         target = as.numeric(targets) * 100) |>
    mutate(
      unweighted = map_dbl(level, ~ mean(df_analysis[[qcol]] == .x, na.rm=TRUE) * 100),
      weighted   = map_dbl(level, ~ sum(df_analysis[[WGT]][df_analysis[[qcol]] == .x],
                                        na.rm=TRUE) / sum(df_analysis[[WGT]]) * 100)
    )
})
print(balance)
write_csv(balance, file.path(OUTPUT_DIR, "weight_balance.csv"))

# ── Balance table → LaTeX ──────────────────────────────────────────────────
balance_fmt <- balance |>
  mutate(across(c(target, unweighted, weighted), ~ round(.x, 1))) |>
  mutate(level = as.character(level))

balance_latex <- c(
  "\\begin{table}[ht]",
  "\\centering",
  "\\small",
  sprintf("\\caption{Raking balance check (RAKE\\_MODE = ``%s''). Values are \\%% of sample.", RAKE_MODE),
  "  Unweighted = raw sample proportions; Weighted = post-raking proportions; Target = YouGov LLM-user benchmarks.}",
  "\\label{tab:weight_balance}",
  "\\begin{tabular}{llccc}",
  "\\toprule",
  "Variable & Level & Unweighted (\\%) & Weighted (\\%) & Target (\\%) \\\\",
  "\\midrule",
  {
    rows <- character(0)
    vars <- unique(balance_fmt$var)
    for (v in vars) {
      sub <- balance_fmt |> filter(var == v)
      for (j in seq_len(nrow(sub))) {
        var_cell <- if (j == 1) v else ""
        rows <- c(rows, sprintf("%s & %s & %.1f & %.1f & %.1f \\\\",
                                var_cell, sub$level[j],
                                sub$unweighted[j], sub$weighted[j], sub$target[j]))
      }
      if (v != tail(vars, 1)) rows <- c(rows, "\\midrule")
    }
    rows
  },
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

cat("\n── Balance table (LaTeX) ─────────────────────────────────────────────────\n")
cat(paste(balance_latex, collapse="\n"), "\n")
writeLines(balance_latex, file.path(OUTPUT_DIR, "weight_balance.tex"))

# ═══════════════════════════════════════════════════════════════════════════
# 2b  Sensitivity: prevalence across weighting methods
# ═══════════════════════════════════════════════════════════════════════════
# 5 configurations: unweighted, union raw/trimmed, intersection raw/trimmed

sens_configs <- list(
  list(label = "Unweighted",              rake_mode = NULL,           trim = FALSE),
  list(label = "Union (raw)",             rake_mode = "union",        trim = FALSE),
  list(label = "Union (trimmed)",         rake_mode = "union",        trim = TRUE),
  list(label = "Intersection (raw)",      rake_mode = "intersection", trim = FALSE),
  list(label = "Intersection (trimmed)",  rake_mode = "intersection", trim = TRUE)
)

build_sens_design <- function(cfg) {
  base <- df_analysis_base
  if (is.null(cfg$rake_mode))
    return(survey::svydesign(ids = ~1, weights = ~1, data = base))

  sel <- all_rake_specs[switch(cfg$rake_mode,
    union        = wt_sel_tbl$gap_crit | wt_sel_tbl$ssl_crit,
    intersection = wt_sel_tbl$gap_crit & wt_sel_tbl$ssl_crit)]
  for (s in sel) base <- dplyr::filter(base, .data[[s$quota_col]] != "Other")

  pop_fn <- function(var, props)
    data.frame(setNames(list(names(props)), var),
               Freq = as.numeric(props) * nrow(base))

  d0    <- survey::svydesign(ids = ~1, weights = ~1, data = base)
  raked <- survey::rake(
    design             = d0,
    sample.margins     = purrr::map(sel, ~ as.formula(paste0("~", .x$quota_col))),
    population.margins = purrr::map(sel, ~ pop_fn(.x$quota_col, .x$targets)),
    control = list(maxit = 500, epsilon = 1e-8, verbose = FALSE))

  w <- weights(raked)
  if (cfg$trim) { w <- cap_weights(w); w <- trim_weights(w) }
  base$weight_sens <- w
  svydesign(ids = ~1, weights = ~weight_sens, data = base)
}

prev_from_design <- function(des) {
  map_dfr(
    c(any_ssl="used_any_ssl", personal="used_personal_ssl",
      conventional="used_conventional_ssl", moral="used_moral_ssl"),
    function(col) {
      est <- svymean(as.formula(paste0("~factor(", col, ")")), des, na.rm=TRUE)
      ci  <- confint(est); idx <- grepl("1$", rownames(ci))
      tibble(estimate=as.numeric(est)[idx], ci_low=ci[idx,1], ci_high=ci[idx,2])
    }, .id = "outcome")
}

cat("\n── Sensitivity: building designs ────────────────────────────────────────\n")
sens_results <- map_dfr(sens_configs, function(cfg) {
  cat(" ", cfg$label, "\n")
  prev_from_design(build_sens_design(cfg)) |> mutate(method = cfg$label)
})

write_csv(sens_results, file.path(OUTPUT_DIR, "prevalence_sensitivity.csv"))

SENS_METHOD_ORDER  <- map_chr(sens_configs, "label")
SENS_OUT_ORDER     <- c("any_ssl","personal","conventional","moral")
SENS_OUT_LABELS    <- c(any_ssl="Any SSL", personal="Personal",
                         conventional="Conventional", moral="Moral")
SENS_OUT_X_LABELS  <- c(any_ssl="Any SSL", personal="Personal",
                         conventional="Conventional\n(Societal)", moral="Moral")

# ── LaTeX table ────────────────────────────────────────────────────────────
sens_latex <- sens_results |>
  mutate(cell = sprintf("%.1f [%.1f, %.1f]",
                        estimate*100, ci_low*100, ci_high*100),
         outcome_label = SENS_OUT_LABELS[outcome],
         outcome_label = factor(outcome_label, levels=SENS_OUT_LABELS)) |>
  select(outcome_label, method, cell) |>
  pivot_wider(names_from=method, values_from=cell) |>
  arrange(outcome_label)

cat("\n── LaTeX sensitivity table ───────────────────────────────────────────────\n")
# build latex manually for clean formatting
col_headers <- paste(SENS_METHOD_ORDER, collapse=" & ")
cat("\\begin{table}[ht]\n")
cat("\\centering\n")
cat("\\small\n")
cat(sprintf("\\caption{SSL prevalence estimates (\\%%) across weighting methods. Values are weighted \\%% [95\\%% CI].}\n"))
cat("\\label{tab:prevalence_sensitivity}\n")
cat(sprintf("\\begin{tabular}{l%s}\n", strrep("c", length(SENS_METHOD_ORDER))))
cat("\\toprule\n")
cat(sprintf("Outcome & %s \\\\\\\\\n", col_headers))
cat("\\midrule\n")
for (i in seq_len(nrow(sens_latex))) {
  row_vals <- as.character(unlist(sens_latex[i, -1]))
  cat(sprintf("%s & %s \\\\\\\\\n",
              sens_latex$outcome_label[i],
              paste(row_vals, collapse=" & ")))
}
cat("\\bottomrule\n")
cat("\\end{tabular}\n")
cat("\\end{table}\n\n")

# Also write to file
latex_lines <- c(
  "\\begin{table}[ht]", "\\centering", "\\small",
  sprintf("\\caption{SSL prevalence estimates (\\%%) across weighting methods. Values are weighted \\%% [95\\%% CI].}"),
  "\\label{tab:prevalence_sensitivity}",
  sprintf("\\begin{tabular}{l%s}", strrep("c", length(SENS_METHOD_ORDER))),
  "\\toprule",
  sprintf("Outcome & %s \\\\", col_headers),
  "\\midrule",
  map_chr(seq_len(nrow(sens_latex)), function(i) {
    row_vals <- as.character(unlist(sens_latex[i, -1]))
    sprintf("%s & %s \\\\", sens_latex$outcome_label[i],
            paste(row_vals, collapse=" & "))
  }),
  "\\bottomrule", "\\end{tabular}", "\\end{table}"
)
writeLines(latex_lines, file.path(OUTPUT_DIR, "prevalence_sensitivity.tex"))

# ── Plot: x = SSL outcome, y = prevalence %, color = method ───────────────
# Color encodes weighting family (grey=unweighted, teal=union, amber=intersection)
# Shape encodes raw vs trimmed (circle=raw/unweighted, triangle=trimmed)
METHOD_COLORS <- c(
  "Unweighted"             = "#888888",
  "Union (raw)"            = PALETTE[1],   # teal
  "Union (trimmed)"        = PALETTE[1],   # teal  (same family, different shape)
  "Intersection (raw)"     = PALETTE[2],   # amber
  "Intersection (trimmed)" = PALETTE[2]    # amber (same family, different shape)
)
METHOD_SHAPES <- c(
  "Unweighted"             = 15,   # filled square
  "Union (raw)"            = 16,   # filled circle
  "Union (trimmed)"        = 17,   # filled triangle
  "Intersection (raw)"     = 1,    # open circle
  "Intersection (trimmed)" = 2     # open triangle
)

p_sens <- sens_results |>
  mutate(method  = factor(method,  levels=SENS_METHOD_ORDER),
         outcome = factor(outcome, levels=SENS_OUT_ORDER)) |>
  ggplot(aes(x=outcome, y=estimate*100,
             color=method, shape=method, group=method)) +
  geom_errorbar(aes(ymin=ci_low*100, ymax=ci_high*100),
                width=0.15, linewidth=0.85,
                position=position_dodge(0.55)) +
  geom_point(size=3.5, position=position_dodge(0.55)) +
  scale_color_manual(values=METHOD_COLORS, name=NULL) +
  scale_shape_manual(values=METHOD_SHAPES, name=NULL) +
  scale_x_discrete(labels=c(any_ssl="Any SSL", personal="Personal",
                             conventional="Conventional", moral="Moral")) +
  scale_y_continuous(limits=c(0,100), labels=function(x) paste0(x,"%"),
                     name="Prevalence (95% CI)") +
  labs(title="SSL Prevalence: Sensitivity to Weighting Method", x=NULL) +
  theme_aesthetic_ggplot(font_scale=1.1) +
  theme(panel.grid.major.x=element_blank(),
        legend.position="bottom",
        legend.text=element_text(size=9))

print(p_sens)
ggsave(file.path(OUTPUT_DIR,"prevalence_sensitivity.pdf"),
       p_sens, width=9, height=6, device="pdf")
ggsave(file.path(OUTPUT_DIR,"prevalence_sensitivity.png"),
       p_sens, width=9, height=6, dpi=300)

# ═══════════════════════════════════════════════════════════════════════════
# 3  Prevalence
# ═══════════════════════════════════════════════════════════════════════════

svy_prop_row <- function(col) {
  # svymean on a binary 0/1 column; returns level==1 estimate
  est <- svymean(as.formula(paste0("~factor(",col,")")), svy, na.rm = TRUE)
  ci  <- confint(est)
  # find the "1" row
  idx <- grepl("1$", rownames(ci))
  tibble(estimate = as.numeric(est)[idx], se = as.numeric(SE(est))[idx],
         ci_low = ci[idx,1], ci_high = ci[idx,2])
}

svy_mean_row <- function(col, design = svy) {
  est <- svymean(as.formula(paste0("~",col)), design, na.rm = TRUE)
  ci  <- confint(est)
  tibble(estimate = as.numeric(est), se = as.numeric(SE(est)),
         ci_low = ci[1], ci_high = ci[2])
}

domain_map <- c(any_ssl                = "used_any_ssl",
                personal_psychological = "used_personal_ssl",
                societal_conventional  = "used_conventional_ssl",
                moral                  = "used_moral_ssl")

prevalence_table <- imap_dfr(domain_map, function(col, label) {
  valid <- df_analysis[[col]]
  est   <- svy_prop_row(col)
  tibble(outcome = label, n_valid = sum(!is.na(valid)), n_used = sum(valid==1,na.rm=TRUE),
         weighted_percent     = round(est$estimate*100, 1),
         weighted_ci_low_pct  = round(est$ci_low*100,  1),
         weighted_ci_high_pct = round(est$ci_high*100, 1),
         weighted_se          = est$se)
})
print(prevalence_table)
write_csv(prevalence_table, file.path(OUTPUT_DIR, "ssl_prevalence.csv"))

# plot
prev_plot_df <- prevalence_table |>
  mutate(label = c("Any SSL","Personal","Conventional","Moral"),
         color = c("#666666", DOMAIN_COLORS[c("personal_psychological",
                                              "societal_conventional","moral")]))

p_prev <- ggplot(prev_plot_df,
                 aes(x = weighted_percent,
                     y = reorder(label, weighted_percent))) +
  geom_col(aes(fill = I(color)), width = 0.55, alpha = 0.88) +
  geom_errorbarh(aes(xmin = weighted_ci_low_pct, xmax = weighted_ci_high_pct),
                 height = 0.15, color = "#333333", linewidth = 0.9) +
  geom_text(aes(x = weighted_ci_high_pct + 2,
                label = paste0(weighted_percent,"%")), hjust=0, size=4.5) +
  scale_x_continuous(limits=c(0,105), labels=function(x) paste0(x,"%")) +
  labs(# title="Synthetic Social Learning (SSL)\nPrevalence Among U.S. Chatbot Users",
       x="Weighted % of respondents (95% CI)", y=NULL) +
  theme_aesthetic_ggplot(font_scale=1.2) +
  theme(panel.grid.major.y = element_blank())

print(p_prev)
ggsave(file.path(OUTPUT_DIR,"prevalence.pdf"), p_prev,
       width=9, height=6, device="pdf")

# frequency distributions
freq_map <- c(personal_psychological="domain_p_freq",
              societal_conventional ="domain_c_freq",
              moral                 ="domain_e_freq")

frequency_table <- imap_dfr(freq_map, function(col, domain) {
  df_analysis |>
    group_by(response = .data[[col]]) |>
    summarise(n = n(),
              weighted_count = sum(.data[[WGT]], na.rm=TRUE), .groups="drop") |>
    mutate(domain           = domain,
           percent          = round(n/sum(n)*100, 1),
           weighted_percent = round(weighted_count/sum(weighted_count)*100, 1))
})
write_csv(frequency_table, file.path(OUTPUT_DIR,"ssl_domain_frequency_distributions.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# 4  Motivations
# ═══════════════════════════════════════════════════════════════════════════

motivation_options <- list(
  practical  = c("Nobody was available at the time",
                 "I needed a response right away",
                 "I can think out loud without needing to pre-organize my thoughts",
                 "I wanted an unbiased opinion"),
  productive = c("I thought using a chatbot would save me time",
                 "I thought using a chatbot would save me effort",
                 "I thought I would learn something",
                 "I thought using a chatbot would produce something valuable"),
  social     = c("I wanted someone to talk to",
                 "I didn’t have a person I could talk to about this in my life",
                 "I wanted AI to provide a particular perspective I didn’t have access to",
                 "I didn't want to burden my friends or family"),
  entertain  = c("I was bored",
                 "I was curious what the chatbot would say",
                 "I was experimenting with different chatbots",
                 "I wanted to be distracted")
)

domain_specs <- list(
  personal_psychological = list(prefix="personal", used_col="used_personal_ssl"),
  societal_conventional  = list(prefix="con",      used_col="used_conventional_ssl"),
  moral                  = list(prefix="moral",    used_col="used_moral_ssl")
)

motivation_crosstab_long <- imap_dfr(domain_specs, function(spec, domain) {
  dom_df <- df_analysis |> filter(.data[[spec$used_col]] == 1)
  imap_dfr(motivation_options, function(options, bucket) {
    map_dfr(options, function(option) {
      col      <- paste0(spec$prefix,"_",bucket)
      col_vals <- replace_na(as.character(dom_df[[col]]), "") |>
        str_replace_all("‘|’|‚", "'")   # normalize curly apostrophes
      option_norm <- str_replace_all(option, "‘|’|‚", "'")
      selected <- str_detect(col_vals, fixed(option_norm))
      w        <- dom_df[[WGT]]
      w_total  <- sum(w, na.rm=TRUE)
      w_sel    <- sum(w[selected], na.rm=TRUE)
      # survey CI
      tmp <- svydesign(ids=~1, weights=~weight_trimmed,
                       data=dom_df |> mutate(sel_val = as.integer(selected)))
      est <- svymean(~sel_val, tmp, na.rm=TRUE)
      ci  <- confint(est)
      tibble(
        domain                           = domain,
        general_bucket                   = bucket,
        specific_factor                  = option,
        n_domain_users                   = nrow(dom_df),
        raw_count                        = sum(selected),
        percent_of_domain_users          = round(mean(selected)*100, 1),
        weighted_count                   = w_sel,
        weighted_percent_of_domain_users = round(as.numeric(est)*100, 1),
        weighted_se                      = as.numeric(SE(est)),
        weighted_ci_low_percent          = round(ci[1]*100, 1),
        weighted_ci_high_percent         = round(ci[2]*100, 1)
      )
    })
  })
})

substantive_crosstab <- motivation_crosstab_long |>
  filter(specific_factor != "None of these reasons applies")

write_csv(motivation_crosstab_long, file.path(OUTPUT_DIR,"ssl_motivation_crosstab_long.csv"))
write_csv(substantive_crosstab,     file.path(OUTPUT_DIR,"rq3_specific_motivation_factors.csv"))

# ── Motivation summary table (rounded to 2 dp) + LaTeX ────────────────────
motiv_tbl <- substantive_crosstab |>
  mutate(
    domain_label = case_when(
      domain == "personal_psychological" ~ "Personal",
      domain == "societal_conventional"  ~ "Conventional",
      domain == "moral"                  ~ "Moral"),
    bucket_label = BUCKET_DISPLAY[general_bucket]
  ) |>
  select(Domain = domain_label, Bucket = bucket_label,
         Motivation = specific_factor,
         `n (domain users)` = n_domain_users,
         `Raw %` = percent_of_domain_users,
         `Wtd. %` = weighted_percent_of_domain_users,
         `Wtd. SE` = weighted_se,
         `95% CI low` = weighted_ci_low_percent,
         `95% CI high` = weighted_ci_high_percent) |>
  mutate(across(where(is.numeric), ~ round(.x, 2))) |>
  arrange(Domain, Bucket, desc(`Wtd. %`))

cat("\n── Motivation table (2 dp) ───────────────────────────────────────────────\n")
print(motiv_tbl, n = Inf)
write_csv(motiv_tbl, file.path(OUTPUT_DIR, "motivation_table_rounded.csv"))

# LaTeX version
motiv_latex_rows <- motiv_tbl |>
  mutate(ci_str = sprintf("[%.2f, %.2f]", `95% CI low`, `95% CI high`)) |>
  select(Domain, Bucket, Motivation, `Wtd. %`, `Wtd. SE`, ci_str)

motiv_latex_lines <- c(
  "\\begin{longtable}{llp{5cm}ccc}",
  "\\toprule",
  "Domain & Bucket & Motivation & Wtd.\\% & SE & 95\\% CI \\\\",
  "\\midrule",
  "\\endfirsthead",
  "\\toprule",
  "Domain & Bucket & Motivation & Wtd.\\% & SE & 95\\% CI \\\\",
  "\\midrule",
  "\\endhead",
  map_chr(seq_len(nrow(motiv_latex_rows)), function(i) {
    r <- motiv_latex_rows[i, ]
    mot <- gsub("&", "\\&", r$Motivation, fixed=TRUE)
    mot <- gsub("%", "\\%", mot, fixed=TRUE)
    sprintf("%s & %s & %s & %.2f & %.2f & %s \\\\",
            r$Domain, r$Bucket, mot,
            r$`Wtd. %`, r$`Wtd. SE`, r$ci_str)
  }),
  "\\bottomrule",
  "\\end{longtable}"
)
writeLines(motiv_latex_lines, file.path(OUTPUT_DIR, "motivation_table.tex"))
cat("LaTeX motivation table written to motivation_table.tex\n")

# RQ1: most common bucket overall
rq1 <- substantive_crosstab |>
  group_by(general_bucket) |>
  summarise(weighted_count = sum(weighted_count), .groups="drop") |>
  mutate(weighted_percent = round(weighted_count/sum(weighted_count)*100,1)) |>
  arrange(desc(weighted_percent))
print(rq1)
write_csv(rq1, file.path(OUTPUT_DIR,"rq1_motivation_bucket_overall.csv"))

# RQ2: by domain
rq2 <- substantive_crosstab |>
  group_by(domain, general_bucket) |>
  summarise(weighted_count = sum(weighted_count), .groups="drop") |>
  group_by(domain) |>
  mutate(weighted_percent_within_domain = round(weighted_count/sum(weighted_count)*100,1)) |>
  ungroup() |> arrange(domain, desc(weighted_percent_within_domain))
print(rq2)
write_csv(rq2, file.path(OUTPUT_DIR,"rq2_motivation_bucket_by_domain.csv"))

# Motivation quadrant plot (2×2 panels, one per bucket)
Y_OFF <- c(personal_psychological=-0.26, societal_conventional=0, moral=0.26)

panel_plots <- map(BUCKET_ORDER, function(bucket) {
  sub <- substantive_crosstab |> filter(general_bucket == bucket)
  factor_ord  <- sub |> group_by(specific_factor) |>
    summarise(m=mean(weighted_percent_of_domain_users),.groups="drop") |>
    arrange(desc(m)) |> pull(specific_factor)

  sub |>
    mutate(specific_factor = factor(specific_factor, levels=factor_ord),
           y_pos = as.numeric(specific_factor) + Y_OFF[domain]) |>
    ggplot(aes(x=weighted_percent_of_domain_users, y=y_pos,
               color=domain, shape=domain)) +
    geom_errorbarh(aes(xmin=weighted_ci_low_percent, xmax=weighted_ci_high_percent),
                   height=0.1, linewidth=0.85, alpha=0.7) +
    geom_point(size=2.5) +
    scale_color_manual(values=DOMAIN_COLORS, labels=DOMAIN_DISPLAY) +
    scale_shape_manual(values=c(personal_psychological=16,
                                societal_conventional =18, moral=17),
                       labels=DOMAIN_DISPLAY) +
    scale_x_continuous(limits=c(0,100), labels=function(x) paste0(x,"%")) +
    scale_y_continuous(breaks=seq_along(factor_ord),
                       labels=str_wrap(factor_ord, 25), trans="reverse") +
    labs(title  = paste0("Motivation: ", BUCKET_DISPLAY[bucket]),
         x      = "Weighted % of Domain Respondents (95% CI)",
         y      = NULL, color="Domain", shape="Domain") +
    theme_aesthetic_ggplot(font_scale=1.1) +
    theme(plot.title   = element_text(size=14, face="italic"),
          axis.title.x = element_text(size=11),
          legend.position = if(bucket=="practical") "bottom" else "none")
})

p_motiv <- wrap_plots(panel_plots, ncol=2) +
  plot_annotation(# title="Motivations by Synthetic Social Learning Domain",
                  theme=theme(plot.title=element_text(size=18, face="bold")))

print(p_motiv)
ggsave(file.path(OUTPUT_DIR,"ssl_motivation_quadrants.pdf"),  p_motiv, width=14, height=12, device="pdf")
ggsave(file.path(OUTPUT_DIR,"ssl_motivation_quadrants.png"),  p_motiv, width=14, height=12, dpi=300)

# ═══════════════════════════════════════════════════════════════════════════
# 4b  Motivation co-occurrence & clustering
# ═══════════════════════════════════════════════════════════════════════════

# Build a binary item matrix: one row per (respondent × domain) observation,
# one column per motivation item. Pool across domains so we have enough power.
normalize_apos <- function(x) str_replace_all(x, "’|’|‚", "'")

all_items <- unlist(motivation_options, use.names=FALSE) |> normalize_apos()
all_items <- all_items[all_items != "None of these reasons applies"]

# Build one row per (respondent × domain), then sum to respondent level
motiv_matrix_long <- imap_dfr(domain_specs, function(spec, domain) {
  dom_df <- df_analysis |> filter(.data[[spec$used_col]] == 1)
  item_cols <- map_dfc(names(motivation_options), function(bucket) {
    col      <- paste0(spec$prefix, "_", bucket)
    col_vals <- normalize_apos(replace_na(as.character(dom_df[[col]]), ""))
    items    <- normalize_apos(motivation_options[[bucket]])
    items    <- items[items != "None of these reasons applies"]
    map_dfc(items, function(item) {
      tibble(!!item := as.integer(str_detect(col_vals, fixed(item))))
    })
  })
  bind_cols(
    tibble(ResponseId = dom_df$ResponseId, domain = domain, weight = dom_df[[WGT]]),
    item_cols
  )
})

# Collapse to one row per respondent: each cell = count of domains mentioning that item (0–3)
# Weight = mean weight across domain rows (weights are respondent-level so should be identical)
# Explicit filter: only respondents who used SSL for at least one domain
motiv_matrix_resp <- motiv_matrix_long |>
  group_by(ResponseId) |>
  summarise(weight = mean(weight),
            across(all_of(all_items), ~ sum(.x, na.rm=TRUE)),
            .groups = "drop") |>
  filter(ResponseId %in% (df_analysis |> filter(used_any_ssl == 1) |> pull(ResponseId)))

# Respondent × item count matrix (one row per respondent)
item_mat <- motiv_matrix_resp |>
  select(all_of(all_items)) |>
  as.matrix()

# ── (a) Jaccard co-occurrence heatmap ──────────────────────────────────────
# Binarize for Jaccard (mentioned in ≥1 domain = 1)
item_mat_bin <- (item_mat > 0) * 1L
jaccard_mat <- matrix(NA_real_, nrow=length(all_items), ncol=length(all_items),
                      dimnames=list(all_items, all_items))
for (i in seq_along(all_items)) {
  for (j in seq_along(all_items)) {
    a <- item_mat_bin[, i]; b <- item_mat_bin[, j]
    inter <- sum(a & b, na.rm=TRUE)
    union <- sum(a | b, na.rm=TRUE)
    jaccard_mat[i, j] <- if (union == 0) 0 else inter / union
  }
}

# Short labels for plot
short_labels <- c(
  "Nobody was available at the time"                                        = "Nobody available",
  "I needed a response right away"                                          = "Needed it fast",
  "I can think out loud without needing to pre-organize my thoughts"        = "Think out loud",
  "I wanted an unbiased opinion"                                            = "Unbiased opinion",
  "I thought using a chatbot would save me time"                            = "Save time",
  "I thought using a chatbot would save me effort"                          = "Save effort",
  "I thought I would learn something"                                       = "Learn something",
  "I thought using a chatbot would produce something valuable"              = "Produce value",
  "I wanted someone to talk to"                                             = "Someone to talk to",
  "I didn't have a person I could talk to about this in my life"            = "No person to ask",
  "I wanted AI to provide a particular perspective I didn't have access to" = "Fresh perspective",
  "I didn't want to burden my friends or family"                            = "Don't burden others",
  "I was bored"                                                             = "Bored",
  "I was curious what the chatbot would say"                                = "Curious",
  "I was experimenting with different chatbots"                             = "Experimenting",
  "I wanted to be distracted"                                               = "Wanted distraction"
)
names(short_labels) <- normalize_apos(names(short_labels))

# Reorder by bucket using all_items (normalized full strings — safe keys)
bucket_order_full <- unlist(lapply(names(motivation_options), function(b) {
  items <- normalize_apos(motivation_options[[b]])
  items[items != "None of these reasons applies"]
}))
jaccard_mat <- jaccard_mat[bucket_order_full, bucket_order_full]

# Now apply short labels (order is now correct)
rownames(jaccard_mat) <- short_labels[rownames(jaccard_mat)]
colnames(jaccard_mat) <- short_labels[colnames(jaccard_mat)]
bucket_order_items    <- rownames(jaccard_mat)   # already in order

jac_df <- as.data.frame(jaccard_mat) |>
  rownames_to_column("item_a") |>
  pivot_longer(-item_a, names_to="item_b", values_to="jaccard") |>
  mutate(item_a = factor(item_a, levels=rev(bucket_order_items)),
         item_b = factor(item_b, levels=bucket_order_items))

# Bucket boundary lines
n_per_bucket <- sapply(motivation_options, function(x) sum(x != "None of these reasons applies"))
bucket_breaks <- cumsum(n_per_bucket) + 0.5
bucket_breaks <- bucket_breaks[-length(bucket_breaks)]

p_jaccard <- ggplot(jac_df, aes(x=item_b, y=item_a, fill=jaccard)) +
  geom_tile(color="white", linewidth=0.4) +
  geom_vline(xintercept=bucket_breaks, color="white", linewidth=1.2) +
  geom_hline(yintercept=length(bucket_order_items) - bucket_breaks + 1, color="white", linewidth=1.2) +
  scale_fill_gradient(low="#f7fbff", high="#00A896",
                      name="Jaccard\nsimilarity", limits=c(0,1)) +
  scale_x_discrete(position="top") +
  labs(x=NULL, y=NULL,
       title=NULL) +
  theme_aesthetic_ggplot(font_scale=0.95) +
  theme(axis.text.x = element_text(angle=45, hjust=0, size=9),
        axis.text.y = element_text(size=9),
        legend.position="right",
        panel.grid = element_blank())

print(p_jaccard)
ggsave(file.path(OUTPUT_DIR,"motivation_jaccard.pdf"), p_jaccard, width=11, height=9, device="pdf")
ggsave(file.path(OUTPUT_DIR,"motivation_jaccard.png"), p_jaccard, width=11, height=9, dpi=300)

# ── (b) Spearman correlation matrix (full, items ordered by hclust) ─────────
# Use raw 0–3 count matrix (number of domains endorsing each item per respondent)
item_mat_cc <- item_mat[complete.cases(item_mat), ]

cor_mat <- cor(item_mat_cc, method="spearman")
colnames(cor_mat) <- short_labels[colnames(cor_mat)]
rownames(cor_mat) <- short_labels[rownames(cor_mat)]

# Reorder items by hierarchical clustering of the correlation matrix
hc       <- hclust(as.dist(1 - cor_mat), method="ward.D2")
hc_order <- hc$labels[hc$order]

cor_ordered <- cor_mat[hc_order, hc_order]

cor_df <- as.data.frame(cor_ordered) |>
  rownames_to_column("item_a") |>
  pivot_longer(-item_a, names_to="item_b", values_to="r") |>
  mutate(item_a = factor(item_a, levels=rev(hc_order)),
         item_b = factor(item_b, levels=hc_order),
         label  = ifelse(item_a == item_b, "", sprintf("%.2f", r)))

p_cor <- ggplot(cor_df, aes(x=item_b, y=item_a, fill=r)) +
  geom_tile(color="white", linewidth=0.4) +
  geom_text(aes(label=label),
            size=2.5,
            color=ifelse(abs(cor_df$r) > 0.35, "white", "gray20")) +
  scale_fill_gradient2(low="#33658A", mid="#f7f7f7", high="#F26419",
                       midpoint=0, limits=c(-1, 1),
                       name="Spearman\nr", na.value="gray90") +
  scale_x_discrete(position="top") +
  labs(x=NULL, y=NULL, title=NULL) +
  theme_aesthetic_ggplot(font_scale=0.95) +
  theme(axis.text.x    = element_text(angle=45, hjust=0, size=9),
        axis.text.y    = element_text(size=9),
        legend.position = "right",
        panel.grid     = element_blank(),
        plot.margin    = margin(t=80, r=10, b=10, l=10))

print(p_cor)
ggsave(file.path(OUTPUT_DIR,"motivation_correlation.pdf"), p_cor, width=11, height=10, device="pdf")
ggsave(file.path(OUTPUT_DIR,"motivation_correlation.png"), p_cor, width=11, height=10, dpi=300)
write_csv(as.data.frame(cor_mat) |> rownames_to_column("item"),
          file.path(OUTPUT_DIR, "motivation_correlation.csv"))

# ── (c) Exploratory factor analysis (oblimin) ────────────────────────────────
# Parallel analysis to select number of factors.
# Method: compares observed eigenvalues to eigenvalues from random data (Monte Carlo)
# and resampled data; suggests retaining factors where observed > simulated.
cat("\n── Parallel analysis for factor selection ──\n")
pa <- psych::fa.parallel(item_mat_cc, fm="ml", fa="fa", plot=FALSE)
cat(sprintf("Parallel analysis suggests %d factor(s)\n", pa$nfact))

N_FACTORS <- pa$nfact

efa <- psych::fa(item_mat_cc, nfactors=N_FACTORS, rotate="oblimin",
                 scores="regression", fm="ml")

cat("\nEFA loadings (pattern matrix, oblimin):\n")
print(psych::fa.sort(efa), digits=2, cutoff=0.25)

# Factor scores — one row per respondent in item_mat_cc, one col per factor
factor_scores <- as.data.frame(efa$scores)
# psych names factors ML1, ML2, ... — keep track of those internal names
ml_names <- colnames(factor_scores)   # e.g. "ML1" "ML3" "ML2" (psych reorders by variance)

# Auto-label each factor by its top 2 loading items (after applying short_labels)
factor_labels <- sapply(ml_names, function(fn) {
  loadings_vec <- abs(efa$loadings[, fn])
  top2         <- names(sort(loadings_vec, decreasing=TRUE)[1:2])
  paste(short_labels[top2], collapse=" / ")
})
names(factor_labels) <- ml_names
cat("\nFactor labels (top 2 items):\n"); print(factor_labels)

# Plain unnamed vector of label strings — safe for all_of() and colnames()
factor_col_names <- unname(factor_labels)

# Rename factor score columns to their auto-labels
colnames(factor_scores) <- factor_col_names

# Loadings heatmap (items × factors, factors in psych's variance order)
load_df <- as.data.frame(unclass(efa$loadings)) |>
  rownames_to_column("item") |>
  pivot_longer(-item, names_to="factor_raw", values_to="loading") |>
  mutate(item   = short_labels[item],
         item   = factor(item, levels=rev(hc_order)),
         factor = factor(factor_labels[factor_raw], levels=factor_col_names))

p_loadings <- ggplot(load_df, aes(x=factor, y=item, fill=loading)) +
  geom_tile(color="white", linewidth=0.4) +
  geom_text(aes(label=sprintf("%.2f", loading)),
            size=2.6,
            color=ifelse(abs(load_df$loading) > 0.35, "white", "gray20")) +
  scale_fill_gradient2(low="#33658A", mid="#f7f7f7", high="#F26419",
                       midpoint=0, limits=c(-1, 1), name="Loading") +
  labs(x=NULL, y=NULL, title=NULL) +
  theme_aesthetic_ggplot(font_scale=0.95) +
  theme(axis.text.x    = element_text(angle=25, hjust=1, size=9),
        axis.text.y    = element_text(size=9),
        legend.position = "right",
        panel.grid     = element_blank())

print(p_loadings)
ggsave(file.path(OUTPUT_DIR,"motivation_efa_loadings.pdf"), p_loadings, width=8, height=9, device="pdf")
ggsave(file.path(OUTPUT_DIR,"motivation_efa_loadings.png"), p_loadings, width=8, height=9, dpi=300)

# ── (d) k-means clustering on continuous motivation counts (0–3) ─────────────
# Each cell: number of domains in which respondent listed that motivation (0–3)
clust_mat <- item_mat_cc   # complete cases, raw counts

# Elbow plot: WSS for k = 1..9
wss_vals <- sapply(1:9, function(k) {
  set.seed(42)
  kmeans(clust_mat, centers=k, nstart=50, iter.max=100)$tot.withinss
})
p_elbow <- ggplot(data.frame(k=1:9, wss=wss_vals), aes(x=k, y=wss)) +
  geom_line(color="#33658A", linewidth=0.8) +
  geom_point(color="#33658A", size=2.5) +
  geom_vline(xintercept=4, linetype="dashed", color="#F26419", linewidth=0.7) +
  scale_x_continuous(breaks=1:9) +
  labs(x="Number of clusters (k)", y="Total within-cluster SS", title="Elbow plot") +
  theme_aesthetic_ggplot(font_scale=1.0)
ggsave(file.path(OUTPUT_DIR, "elbow_plot.pdf"), p_elbow, width=5, height=3.5, device="pdf")
ggsave(file.path(OUTPUT_DIR, "elbow_plot.png"), p_elbow, width=5, height=3.5, dpi=300)

best_k <- 4
cat(sprintf("\nUsing k=%d clusters (continuous motivation counts)\n", best_k))

set.seed(42)
km <- kmeans(clust_mat, centers=best_k, nstart=50, iter.max=100)

clust_clustered <- as_tibble(clust_mat) |>
  dplyr::mutate(cluster = km$cluster)

# Heatmap: mean count per item per cluster
cluster_means <- clust_clustered |>
  dplyr::group_by(cluster) |>
  dplyr::summarise(dplyr::across(dplyr::all_of(all_items), mean), n=dplyr::n(), .groups="drop") |>
  dplyr::arrange(cluster)

cat("\nCluster sizes:\n"); print(cluster_means |> dplyr::select(cluster, n))

cluster_hm_df <- cluster_means |>
  tidyr::pivot_longer(dplyr::all_of(all_items), names_to="item", values_to="avg") |>
  dplyr::mutate(
    item        = short_labels[item],
    item        = factor(item, levels=short_labels[all_items]),
    cluster_lbl = factor(
      paste0("Cluster ", cluster, "\n(n=", cluster_means$n[cluster], ")"),
      levels = paste0("Cluster ", seq_len(best_k), "\n(n=", cluster_means$n, ")")
    )
  )

n_items <- length(all_items)
hm_h    <- max(5, n_items * 0.28)

p_cluster_hm <- ggplot(cluster_hm_df, aes(x=cluster_lbl, y=item, fill=avg)) +
  geom_tile(color="white", linewidth=0.5) +
  geom_text(aes(label=sprintf("%.2f", avg)),
            size=2.8, fontface="bold",
            color=ifelse(cluster_hm_df$avg > 1.5, "white", "gray20")) +
  scale_fill_gradient(low="#f7f7f7", high="#33658A", limits=c(0, 3),
                      name="Mean count\n(0–3)") +
  labs(x=NULL, y=NULL, title=NULL) +
  theme_aesthetic_ggplot(font_scale=1.0) +
  theme(axis.text.x     = element_text(size=10),
        axis.text.y     = element_text(size=8),
        legend.position = "right",
        panel.grid      = element_blank())

print(p_cluster_hm)
ggsave(file.path(OUTPUT_DIR,"motivation_clusters.pdf"), p_cluster_hm,
       width=8, height=hm_h, device="pdf")
ggsave(file.path(OUTPUT_DIR,"motivation_clusters.png"), p_cluster_hm,
       width=8, height=hm_h, dpi=300)

write_csv(cluster_means, file.path(OUTPUT_DIR, "motivation_cluster_profiles.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# 5  Outcomes (satisfaction & recommendation)
# ═══════════════════════════════════════════════════════════════════════════

df_analysis <- df_analysis |>
  mutate(
    personal_satisfied_num  = extract_leading_number(personal_satisfied),
    personal_recfriend_num  = extract_leading_number(personal_recfriend),
    con_satisfied_num       = extract_leading_number(con_satisfied),
    con_recfriend_num       = extract_leading_number(con_recfriend),
    moral_satisfied_num     = extract_leading_number(moral_satisfied),
    moral_recfriend_num     = extract_leading_number(moral_recfriend)
  )

# rebuild svy after adding columns
svy <- svydesign(ids=~1, weights=as.formula(paste0("~",WGT)), data=df_analysis)

sat_rec_specs <- list(
  personal_psychological = list(used="used_personal_ssl",
                                satisfied="personal_satisfied_num",
                                recfriend="personal_recfriend_num"),
  societal_conventional  = list(used="used_conventional_ssl",
                                satisfied="con_satisfied_num",
                                recfriend="con_recfriend_num"),
  moral                  = list(used="used_moral_ssl",
                                satisfied="moral_satisfied_num",
                                recfriend="moral_recfriend_num")
)

sat_rec_df <- imap_dfr(sat_rec_specs, function(spec, domain) {
  dom_df <- df_analysis |> filter(.data[[spec$used]] == 1)
  imap_dfr(list(satisfied=spec$satisfied, recfriend=spec$recfriend), function(col, metric) {
    tmp <- svydesign(ids=~1, weights=~weight_trimmed,
                     data=dom_df |> filter(!is.na(.data[[col]])))
    est <- svymean(as.formula(paste0("~",col)), tmp, na.rm=TRUE)
    ci  <- confint(est)
    tibble(domain=domain, metric=metric, n=nrow(dom_df |> filter(!is.na(.data[[col]]))),
           estimate=as.numeric(est), se=as.numeric(SE(est)),
           ci_low=ci[1], ci_high=ci[2])
  })
})
print(sat_rec_df)
write_csv(sat_rec_df, file.path(OUTPUT_DIR,"sat_rec.csv"))

# plot
# PALETTE[1] = teal (#00A896), PALETTE[2] = amber (#F6AE2D)
METRIC_COLORS <- c(satisfied = PALETTE[1], recfriend = PALETTE[2])
METRIC_OFF    <- c(satisfied=-0.18, recfriend=0.18)
METRIC_LABELS <- c(satisfied = "Satisfied with experience",
                   recfriend = "Would recommend a friend use chatbot this way")

p_outcomes <- sat_rec_df |>
  mutate(domain_label = DOMAIN_DISPLAY[domain],
         y_base       = as.numeric(factor(domain_label, levels=DOMAIN_DISPLAY)),
         y_pos        = y_base + METRIC_OFF[metric],
         metric       = factor(metric, levels=c("satisfied","recfriend"))) |>
  ggplot(aes(x=estimate-1, y=y_pos, fill=metric, color=metric)) +
  geom_col(width=0.28, alpha=0.85, position="identity", orientation="y") +
  geom_errorbar(aes(xmin=ci_low-1, xmax=ci_high-1),
                width=0.07, linewidth=0.9, orientation="y") +
  geom_text(aes(x=ci_high-1+0.12, label=paste0("M=", round(estimate,1))),
            hjust=0, size=3.8, color="#2C3531", show.legend=FALSE) +
  scale_fill_manual(values=METRIC_COLORS, labels=METRIC_LABELS, name=NULL) +
  scale_color_manual(values=METRIC_COLORS, labels=METRIC_LABELS, name=NULL) +
  scale_x_continuous(limits=c(0,9), breaks=0:8,
                     labels=function(x) x+1, name="Score (1-9 scale)") +
  scale_y_continuous(breaks=seq_along(DOMAIN_DISPLAY),
                     labels=DOMAIN_DISPLAY, name=NULL) +
  labs(# title="SSL Experience: Satisfaction & Recommendation"
  ) +
  theme_aesthetic_ggplot(font_scale=1.2) +
  theme(panel.grid.major.y=element_blank(),
        legend.position="bottom",
        legend.text=element_text(size=10))

print(p_outcomes)
ggsave(file.path(OUTPUT_DIR,"outcomes_sat_rec.pdf"), p_outcomes,
       width=9, height=6, device="pdf")

# ── SSL intensity ──────────────────────────────────────────────────────────
df_analysis <- df_analysis |>
  mutate(
    personal_ssl_intensity              = frequency_code(domain_p_freq),
    societal_conventional_ssl_intensity = frequency_code(domain_c_freq),
    moral_ssl_intensity                 = frequency_code(domain_e_freq),
    ssl_intensity_total                 = personal_ssl_intensity +
                                          societal_conventional_ssl_intensity +
                                          moral_ssl_intensity
  )

# rebuild svy one final time
svy <- svydesign(ids=~1, weights=as.formula(paste0("~",WGT)), data=df_analysis)

# ═══════════════════════════════════════════════════════════════════════════
# 6  Regressions
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PREDS <- c(
  "age_num_z","gender_male","gender_other","race_white","race_black",
  "pid_republican","pid_democrat","income_ord_z","college_degree",
  "employed","retired","ideo_num_z",
  "use_freq_code_z","aias4_score_z","anthrotech_score_z",
  "sias4_score_z","lsns6_score_z",
  "tipi_extraversion_z","tipi_agreeableness_z","tipi_conscientiousness_z",
  "tipi_emotional_stability_z","tipi_openness_z","ai_social_use_index_z"
)

PREDICTOR_LABELS <- c(
  age_num_z                  = "Age (z)",
  gender_male                = "Male  [ref: Female]",
  gender_other               = "Gender: other  [ref: Female]",
  race_white                 = "White  [ref: other race]",
  race_black                 = "Black  [ref: other race]",
  pid_republican             = "Republican  [ref: Independent]",
  pid_democrat               = "Democrat  [ref: Independent]",
  income_ord_z               = "Income (z)",
  college_degree             = "College degree",
  employed                   = "Employed  [ref: not]",
  retired                    = "Retired  [ref: not]",
  ideo_num_z                 = "Ideology: conservative (z)",
  use_freq_code_z            = "LLM use frequency (z)",
  aias4_score_z              = "AI attitudes (z)",
  anthrotech_score_z         = "AI anthropomorphism (z)",
  sias4_score_z              = "Social anxiety (z)",
  lsns6_score_z              = "Social network size (z)",
  tipi_extraversion_z        = "Extraversion (z)",
  tipi_agreeableness_z       = "Agreeableness (z)",
  tipi_conscientiousness_z   = "Conscientiousness (z)",
  tipi_emotional_stability_z = "Emotional stability (z)",
  tipi_openness_z            = "Openness (z)",
  ai_social_use_index_z      = "AI social use index (z)"
)
label_pred <- function(t) { l <- PREDICTOR_LABELS[t]; if(is.na(l)) t else unname(l) }

LOGIT_OUTCOMES <- c(any="used_any_ssl", personal="used_personal_ssl",
                    conventional="used_conventional_ssl", moral="used_moral_ssl")
OLS_OUTCOMES   <- c(total_intensity="ssl_intensity_total",
                    personal_intensity="personal_ssl_intensity",
                    conventional_intensity="societal_conventional_ssl_intensity",
                    moral_intensity="moral_ssl_intensity")
OUTCOME_LABELS <- c(any="Any SSL", personal="Any Personal SSL",
                    conventional="Any Conventional SSL", moral="Any Moral SSL",
                    total_intensity="SSL Intensity", personal_intensity="Personal SSL Intensity",
                    conventional_intensity="Conventional SSL Intensity",
                    moral_intensity="Moral SSL Intensity")

reg_fml <- function(outcome) as.formula(paste(outcome,"~",paste(MODEL_PREDS,collapse="+")))

# ── VIF ────────────────────────────────────────────────────────────────────
reg_df <- df_analysis |>
  select(all_of(c(MODEL_PREDS, unname(c(LOGIT_OUTCOMES,OLS_OUTCOMES)), WGT))) |>
  mutate(across(everything(), as.numeric)) |> drop_na()

vif_tbl <- tibble(term      = names(vif(lm(reg_fml("used_any_ssl"), data=reg_df))),
                  predictor = map_chr(names(vif(lm(reg_fml("used_any_ssl"), data=reg_df))), label_pred),
                  vif       = as.numeric(vif(lm(reg_fml("used_any_ssl"), data=reg_df)))) |>
  arrange(desc(vif))
print(vif_tbl)
write_csv(vif_tbl, file.path(OUTPUT_DIR,"regression_predictor_vifs.csv"))

# ── unweighted + weighted Spearman ────────────────────────────────────────
eff_n       <- function(w) sum(w)^2 / sum(w^2)
fisher_ci   <- function(r, n, z=1.96) { r <- pmax(pmin(r,1-1e-10),-1+1e-10)
                                         se <- 1/sqrt(n-3); zr <- atanh(r)
                                         c(tanh(zr-z*se), tanh(zr+z*se)) }
fisher_p    <- function(r, n) { r <- pmax(pmin(r,1-1e-10),-1+1e-10)
                                  2*pnorm(-abs(atanh(r)*sqrt(n-3))) }

spearman_table <- imap_dfr(c(LOGIT_OUTCOMES, OLS_OUTCOMES), function(ocol, olbl) {
  map_dfr(MODEL_PREDS, function(pred) {
    d <- reg_df |> select(all_of(c(ocol, pred, WGT))) |> drop_na()
    if (nrow(d) < 4) {
      empty <- tibble(outcome=olbl, outcome_label=OUTCOME_LABELS[olbl], term=pred,
                      predictor=label_pred(pred), model=NA_character_,
                      spearman_r=NA, ci_low=NA, ci_high=NA, p=NA, n=nrow(d), effective_n=NA)
      return(bind_rows(empty, empty |> mutate(model="weighted")))
    }
    # unweighted
    r_uw <- cor(d[[pred]], d[[ocol]], method="spearman")
    n_uw <- nrow(d)
    ci_uw <- fisher_ci(r_uw, n_uw)
    row_uw <- tibble(outcome=olbl, outcome_label=OUTCOME_LABELS[olbl], term=pred,
                     predictor=label_pred(pred), model="unweighted",
                     spearman_r=r_uw, ci_low=ci_uw[1], ci_high=ci_uw[2],
                     p=fisher_p(r_uw, n_uw), n=n_uw, effective_n=n_uw)
    # weighted
    r_wt <- weightedCorr(d[[pred]], d[[ocol]], method="Spearman", weights=d[[WGT]])
    ne   <- eff_n(d[[WGT]])
    ci_wt <- fisher_ci(r_wt, ne)
    row_wt <- tibble(outcome=olbl, outcome_label=OUTCOME_LABELS[olbl], term=pred,
                     predictor=label_pred(pred), model="weighted",
                     spearman_r=r_wt, ci_low=ci_wt[1], ci_high=ci_wt[2],
                     p=fisher_p(r_wt, ne), n=n_uw, effective_n=ne)
    bind_rows(row_uw, row_wt)
  })
})

cat("\n── Zero-order Spearman correlations ─────────────────────────────────────\n")
print(spearman_table |>
        select(outcome_label, predictor, model, spearman_r, ci_low, ci_high, p, n) |>
        arrange(outcome_label, predictor, model), n = Inf)
write_csv(spearman_table, file.path(OUTPUT_DIR,"regression_zero_order_spearman.csv"))

# ── Significant Spearman rhos — copy-pasteable LaTeX strings ──────────────
cat("\n── Significant Spearman rhos (p < .05) — paste into Python/LaTeX ─────────\n")
spearman_table |>
  filter(!is.na(p), p < .05) |>
  arrange(outcome_label, model, p) |>
  rowwise() |>
  mutate(latex_str = sprintf(
    '"%s | %s | %s":  r"$\\rho = %.2f$, 95\\%% CI $[%.2f,\\, %.2f]$, $p %s$"',
    outcome_label, predictor, model,
    spearman_r, ci_low, ci_high,
    if (p < .001) "< .001" else sprintf("= %.3f", p)
  )) |>
  pull(latex_str) |>
  cat(sep = "\n")
cat("\n")

# Spearman forest plots — unweighted + weighted dodged (intensity outcomes only)
walk(names(OLS_OUTCOMES), function(olbl) {
  pd <- spearman_table |>
    filter(outcome == olbl, !is.na(model)) |>
    group_by(predictor) |>
    mutate(mean_r = mean(spearman_r, na.rm=TRUE)) |> ungroup() |>
    mutate(predictor = factor(predictor, levels = unique(predictor[order(mean_r)])),
           sig       = !is.na(p) & p < .05,
           dot_color = case_when(sig & spearman_r >  0 ~ PALETTE[1],
                                 sig & spearman_r <= 0 ~ PALETTE[7],
                                 TRUE                  ~ "#aaaaaa"),
           model     = factor(model, levels = c("unweighted","weighted")))
  p <- ggplot(pd, aes(x=spearman_r, y=predictor,
                      color=I(dot_color), shape=model)) +
    geom_vline(xintercept=0, color="#cccccc", linewidth=0.8) +
    geom_errorbarh(aes(xmin=ci_low, xmax=ci_high),
                   height=0.2, linewidth=0.8,
                   position=position_dodge(0.5)) +
    geom_point(size=2.5, position=position_dodge(0.5)) +
    scale_shape_manual(values=c(unweighted=16, weighted=18),
                       name=NULL) +
    scale_x_continuous(limits=c(-1,1),
                       name="Zero-order Spearman r (95% CI)") +
    labs(title=OUTCOME_LABELS[olbl], y=NULL) +
    theme_aesthetic_ggplot(font_scale=1.1) +
    theme(panel.grid.major.y=element_blank(),
          legend.position="bottom")
  print(p)
  ggsave(file.path(OUTPUT_DIR, paste0("zero_order_spearman_",olbl,".pdf")),
         p, width=9, height=8, device="pdf")
})

# ── fit all regression models ──────────────────────────────────────────────
fit_models <- function(outcome_col, label, family) {
  dat <- df_analysis |>
    select(all_of(c(outcome_col, MODEL_PREDS, WGT))) |>
    mutate(across(everything(), as.numeric)) |> drop_na()

  svy_d <- svydesign(ids=~1, weights=~weight_trimmed, data=dat)

  uw <- tidy(glm(reg_fml(outcome_col), data=dat, family=family), conf.int=TRUE) |>
    mutate(outcome=label, model="unweighted", n=nrow(dat))
  wt <- tidy(svyglm(reg_fml(outcome_col), design=svy_d, family=family), conf.int=TRUE) |>
    mutate(outcome=label, model="weighted", n=nrow(dat))
  list(uw=uw, wt=wt)
}

logit_results <- imap(LOGIT_OUTCOMES, function(col,lbl) {
  cat("logit:", lbl, "\n"); fit_models(col, lbl, binomial) })
ols_results   <- imap(OLS_OUTCOMES,   function(col,lbl) {
  cat("OLS:",   lbl, "\n"); fit_models(col, lbl, gaussian) })

all_logit <- bind_rows(map(logit_results, ~ bind_rows(.x$uw, .x$wt)))
all_ols   <- bind_rows(map(ols_results,   ~ bind_rows(.x$uw, .x$wt)))
cat("\n── Logit regression results ─────────────────────────────────────────────\n")
print(all_logit |> filter(term != "(Intercept)") |>
        select(outcome, model, term, estimate, std.error, p.value, conf.low, conf.high) |>
        arrange(outcome, model, p.value), n = Inf)

cat("\n── OLS regression results ───────────────────────────────────────────────\n")
print(all_ols |> filter(term != "(Intercept)") |>
        select(outcome, model, term, estimate, std.error, p.value, conf.low, conf.high) |>
        arrange(outcome, model, p.value), n = Inf)

write_csv(all_logit, file.path(OUTPUT_DIR,"regression_logit.csv"))
write_csv(all_ols,   file.path(OUTPUT_DIR,"regression_ols.csv"))

# ── forest plots ───────────────────────────────────────────────────────────
forest_plot <- function(tbl_list, lbl, exp=FALSE) {
  tbl <- bind_rows(tbl_list[[lbl]]$uw, tbl_list[[lbl]]$wt) |>
    filter(term != "(Intercept)") |>
    mutate(
      predictor  = map_chr(term, label_pred),
      cv         = if(exp) exp(estimate)  else estimate,
      clo        = if(exp) exp(conf.low)  else conf.low,
      chi        = if(exp) exp(conf.high) else conf.high,
      thr        = if(exp) 1 else 0,
      sig        = !is.na(p.value) & p.value < .05,
      dot_color  = case_when(sig & cv>thr ~ PALETTE[1],
                             sig & cv<thr ~ PALETTE[7], TRUE ~ "#aaaaaa")
    ) |>
    group_by(predictor) |> mutate(m=mean(abs(cv-thr))) |> ungroup() |>
    arrange(m) |> mutate(predictor=fct_inorder(predictor))

  ggplot(tbl, aes(x=cv, y=predictor, color=I(dot_color), shape=model)) +
    geom_vline(xintercept=if(exp) 1 else 0,
               color="#cccccc", linetype="dashed", linewidth=0.7) +
    geom_errorbarh(aes(xmin=clo, xmax=chi),
                   height=0.2, linewidth=0.85, position=position_dodge(0.4)) +
    geom_point(size=3, position=position_dodge(0.4)) +
    scale_shape_manual(values=c(unweighted=16, weighted=18)) +
    labs(title=OUTCOME_LABELS[lbl], shape=NULL,
         x=if(exp) "Odds ratio (95% CI)" else "Coefficient (95% CI)", y=NULL) +
    theme_aesthetic_ggplot(font_scale=1.0) +
    theme(legend.position="bottom", panel.grid.major.y=element_blank())
}

for (lbl in names(LOGIT_OUTCOMES)) {
  p <- forest_plot(logit_results, lbl, exp=TRUE)
  print(p)
  ggsave(file.path(OUTPUT_DIR, paste0("logit_or_",lbl,".pdf")),
         p, width=9, height=8, device="pdf")
}
for (lbl in names(OLS_OUTCOMES)) {
  p <- forest_plot(ols_results, lbl, exp=FALSE)
  print(p)
  ggsave(file.path(OUTPUT_DIR, paste0("ols_coef_",lbl,".pdf")),
         p, width=9, height=8, device="pdf")
}

cat("\nDone. Outputs in:", OUTPUT_DIR, "\n")
