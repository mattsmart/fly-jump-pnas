# Script for Data Analysis and Visualization
# Related to: "Dynamical modeling of individual sensory reactivity and habituation learning" 
# Author: Marina Boon

# This script performs data wrangling, statistical analysis, 
# and generates Figures 1E, 4C,D,I, S1, S6, S7, and S8 

#---------------------------Instructions for Use---------------------------

# 1. R Version:
#This script was developed and tested using R version 4.4.1. 
#Compatibility with earlier versions is not guaranteed.

# 2. Required Packages:
#    The script uses the following R packages:
#    tidyverse, grid, car, zoo, lme4, lmerTest, ggpubr, psych, rptR, emmeans, boot, ggh4x
#    If not already installed, install them using install.packages("<package_name>").
#    You may also define the library path using .libPaths() if needed.

# 3. Working Directory:
#    Set the working directory to the folder containing the data files using setwd().
#    Example:
#    setwd("/path/to/your/folder")

# 4. Input Files:
#    Ensure the following input files are present in the working directory: 
#     data_raw.csv, data_bay1.csv and data_bay2.csv.
#   - data_raw.csv contains unprocessed behavioral data from all experimental trials. 
#     Columns include: 
#       - genetic_background: the genetic background of the fly. 
#         GD: the genetic background used to establish the GD RNAi library
#         KK: the genetic background used to establish the KK RNAi library
#       - exp_id: experiment identifier, 
#       - chamber: experimental chamber identifier, 
#       - box: experimental box identifier,
#       - assay: type of assay, with the following possible values: 
#         - accl = acclimatization phase, 
#         - hab1 to hab5 = habituation assays 1 through 5, and 
#         - sra = (stimulus) reactivity assay, 
#       - age: age of the fly at testing, in days post-eclosion, 
#       - fly_id: unique identifier for each fly,
#       - trial: for habituation and reactivity assays, this variable represents the sequential
#         trial number within that specific assay, for the acclimatization phase, it represents
#         the second (1-300) of the 5-minute period, with one row per second.
#       - jump: binary response variable of the behavioral assay, where 
#         - 1 = jump (sound) recorded, and 
#         - 0 = no jump recorded.
#   - data_bay1.csv and data_bay2.csv contain posterior samples of individual-specific parameters inferred from 
#     a Bayesian model of fly jump behavior. For each fly, the dataset includes 800 posterior
#     draws of the parameter triplet (α, β, p₀) and derived quantities of interest.
#     Columns include:
#       - genetic_background: the genetic background of the fly. 
#         GD: the genetic background used to establish the GD RNAi library
#         KK: the genetic background used to establish the KK RNAi library
#       - age: age of the fly at testing, in days post-eclosion, 
#       - fly_id: unique identifier for each fly,
#       - draw: sequential posterior draw number,
#       - alpha: predicted decay rate of the memory state
#       - beta: predicted accumulation rate of the memory state during stimulation
#       - k_star: predicted habituation half time; the first trial at which the 
#         predicted jump rate crosses the midpoint between p_0 and p_ss
#       - m_abs: predicted absolute habituation magnitude (p_0 - p_ss)
#       - m_rel: predicted relative habituation magnitude (1 - p_ss/p_0)
#       - p_ss: predicted steady-state jump rate
#       - ttc: predicted number of trials before reaching a criterion of 5 consecutive non-jumps
#       - p_0: predicted jump rate at trial 0
#       - p_reactivity: predicted jump rate in the reactivity trials
#       - chamber: experimental chamber identifier.

# 5. Running the Script:
#    Run the entire script from top to bottom. It is organized into sections:
#   - Data loading and calculating empirical data statistics per fly per testing day:
#     - ttc_emp: the Trials To Criterion (TTC); for each habituation assay, the last trial to
#       which the fly jumped before it stopped jumping for five consecutive trials,
#     - p_ss_emp: for each habituation assay, the jump rate in trials 50 through 200,
#     - p_transient_emp: for each habituation assay, the jump rate in trials 1 through 10,
#     - p_reactivity_emp: the jump rate in the reactivity assay,
#     - m_abs_emp: for each habituation assay, the absolute habituation magnitude 
#       (p_reactivity_emp - p_ss_emp),
#     - m_rel_emp: for each habituation assay, the relative habituation magnitude 
#       (1 - p_ss_emp/p_reactivity_emp),
#     - k_star_emp: the habituation half-time; for each habituation assay, the first trial at which
#       the average jump rate per five trials is equal to or lower than 
#       p_reactivity_emp - 0.5 * m_abs_emp. 
#   - Assessment of spontaneous jumping in the acclimatization phase
#   - Computing average jump response curves and graphing figure 1E and S7A
#   - Assessment of population-level jump rate in trials 1-10 per habituation assay, 
#       and per reactivity block of 10 trials, including graphing figure S1A,B
#   - Mean Pairwise Squared Differences analyses and graphing figures S6 and S8A
#   - Repeatability analyses and graphing figures 4C,D and S8B,C
#   - Assessment of distribution of sample standard deviations of the mean p_ss and p_0 per fly
#       across ages, including graphing figure 4I

# 6. Output:
#    The script produces the following outputs:
#   - Summary statistics and results of repeatabiltiy analyses printed to the console
#   - Figures displayed in the plot window, and saved as .pdf files in the working directory
#   - Results of the repeatability analyses within testing day and across ages for KK and GD 
#       separately; printed to the console and saved in csv files in the working directory. 
#       See https://cran.r-project.org/web/packages/rptR/vignettes/rptR.html for more
#       information on repeatability analyses using the rptR package.

#---------------------------Begin Script---------------------------

library(tidyverse)
library(grid)
library(car)
library(zoo)
library(lme4)
library(lmerTest)
library(ggpubr)
library(psych)
library(rptR) 
library(emmeans)
library(boot) 
library(ggh4x)

data_raw <- read.csv("data_raw.csv")
data_bay1 <- read.csv("data_bay1.csv") 
data_bay2 <- read.csv("data_bay2.csv") 
data_bay <- bind_rows(data_bay1, data_bay2)

ages <- c(7, 14, 21)
genetic_backgrounds <- c("GD", "KK")

#Calculating empirical data statistics and creating data frames ----
##data_per_fly ----
data_raw <- as_tibble(data_raw)
data_raw <- data_raw %>% 
  mutate(across(c(genetic_background, chamber, box, assay, fly_id), as.factor),
         across(c(age, trial, jump), as.numeric))

data_per_fly <- data_raw %>% 
  group_by(genetic_background, age, assay, fly_id) %>% 
  mutate(p_5 = rollmean(jump, k = 5, fill = NA, align = "left"))

#function to calculate streaks of non-jumping, to calculate TTC
calculate_stop_streak <- function(jump) {
  stop_streak <- integer(length(jump))
  streak <- 0
  for (i in seq_along(jump)) {
    if (jump[i] == 0) {
      streak <- streak + 1
    } else {
      streak <- 0
    }
    stop_streak[i] <- streak
  }
  return(stop_streak)
}

data_per_fly <- data_per_fly %>%
  group_by(genetic_background, age, assay, fly_id, chamber, box) %>%
  arrange(genetic_background, age, assay, fly_id, trial) %>%  
  mutate(
    stop_streak = calculate_stop_streak(jump),  
    stop_streak_5 = ifelse(stop_streak == 5, trial-5, NA), 
    ttc_emp = if (all(is.na(stop_streak_5))) max(trial, na.rm = TRUE) 
      else min(stop_streak_5, na.rm = TRUE),  
    p_ss_emp = if_else(str_detect(assay, "hab"), mean(jump[trial >= 50 & trial <= 200]),
                       NA_real_),  
    p_transient_emp = if_else(str_detect(assay, "hab"), mean(jump[trial <= 10]), NA_real_)  ) %>%  
  ungroup() 

#calculate jump rate in the stimulus reactivity assay (sra) and add to data_per_fly
data_per_fly <- left_join(data_per_fly, data_raw %>% 
                            filter(assay == "sra") %>%
                            group_by(genetic_background, age, fly_id, chamber, box) %>%
                            summarise(p_reactivity_emp = mean(jump, na.rm = TRUE)))

data_per_fly <- data_per_fly %>% 
  group_by(genetic_background, age, assay, fly_id) %>%
  mutate(
    m_abs_emp = if_else(str_detect(assay, "hab"), p_reactivity_emp - p_ss_emp, NA_real_),  
    m_rel_emp = if_else(str_detect(assay, "hab") & p_reactivity_emp >0, 
                        1 - p_ss_emp / p_reactivity_emp, NA_real_),  
    half_time_threshold = case_when(
      m_abs_emp > 0 ~ p_reactivity_emp - 0.5*m_abs_emp,
      TRUE ~ NA), 
    k_star_emp = trial[which(p_5 <= half_time_threshold)[1]] 
    ) %>%
  select(genetic_background, exp_id, age, assay, fly_id, chamber, box, ttc_emp, p_transient_emp, p_ss_emp, p_reactivity_emp, 
         m_abs_emp, m_rel_emp, k_star_emp) %>% 
  arrange(genetic_background, age, assay, fly_id) %>%  
  distinct(genetic_background, age, assay, fly_id, .keep_all = TRUE)  


##data_sra_block----
data_sra_block <- data_raw %>% 
  filter(assay == "sra") %>% 
  group_by(genetic_background, age, fly_id, chamber, box) %>%
  arrange(genetic_background, age, fly_id, trial) %>%  
  mutate(block = floor((trial - 1) / 10) + 1) %>%  
  group_by(genetic_background, age, block, fly_id, chamber, box) %>% 
  summarise(p_reactivity_block_emp = sum(jump)/10) %>% 
  ungroup() 


##data_stats_summary ----
data_stats_summary <- data_per_fly %>% 
  filter(str_detect(assay, "hab|sra")) %>% 
  group_by(genetic_background, age, chamber, box, fly_id) %>% 
  summarise(across(where(is.numeric), 
                   list(mean = ~ mean(.x, na.rm = TRUE)),
                   .names = "{fn}_{.col}"), 
            .groups = "drop") %>% 
  ungroup() %>%
  mutate(across(where(is.numeric), ~ ifelse(is.nan(.), NA, .))) 

#Replace the variables with mean jump rate from SRA in data_stats_summary, 
# with the meaningful values from the data_sra_block dataframe  
data_stats_sra_block_summary <- data_sra_block %>%
  group_by(genetic_background, age, fly_id) %>%
  summarize(mean_p_reactivity_emp = mean(p_reactivity_block_emp, na.rm = TRUE))

data_stats_summary <- data_stats_summary %>%
  left_join(data_stats_sra_block_summary, by = c("genetic_background", "age", "fly_id")) %>%
  mutate(mean_p_reactivity_emp = coalesce(mean_p_reactivity_emp.y, mean_p_reactivity_emp.x)) %>%
  select(-contains(".x"), -contains(".y"))


##transformations ----
#function to perform monotonic log-odds transformations
# using a suitable epsilon for each parameter
logit_transform_monotonic <- function(data, columns) {
  valid_columns <- intersect(columns, names(data))
  invalid_columns <- setdiff(columns, names(data))
  if (length(invalid_columns) > 0) {
    warning("The following columns were not found in the data and will be skipped: ", 
            paste(invalid_columns, collapse = ", "))
  }
  for (col in columns) {
    values <- data[[col]]
    if (all(values >= 0 & values <= 1, na.rm = TRUE)) {
      min_nonzero <- min(values[values > 0], na.rm = TRUE)
      max_less_than_one <- max(values[values < 1], na.rm = TRUE)
      epsilon <- 2 * min(min_nonzero, 1 - max_less_than_one)
      message("Epsilon used for ", col, ": ", epsilon)
      adjusted <- (1 - epsilon) * values + epsilon / 2
      data[[paste0(col, "_logit")]] <- logit(adjusted)
    } else {
      warning("Column ", col, " contains values outside [0, 1] and was skipped.")
    }
  }
  return(data)
}


data_stats <- data_per_fly %>% 
  mutate(across(c(ttc_emp, k_star_emp), ~ log(. + 1), .names = "{.col}_log")) %>%
  logit_transform_monotonic(columns = c("p_transient_emp", "p_ss_emp", "p_reactivity_emp")
  )

data_stats_sra_block <- data_sra_block %>% 
  mutate(age_factor = as.factor(age),
         block_factor = as.factor(block)) %>% 
  logit_transform_monotonic(columns = c("p_reactivity_block_emp")
  )

##data_bay_summary ----
data_bay <- as_tibble(data_bay)
data_bay <- data_bay %>% 
  mutate(across(c(genetic_background, fly_id), as.factor),
         across(c(age, draw, alpha, beta, k_star, m_abs, m_rel, p_ss, ttc, p_0, p_reactivity), 
                as.numeric))
data_bay <- left_join(data_bay, data_stats %>% filter(assay == "sra") %>%  
                        select(genetic_background, age, fly_id, chamber)) #to add the chamber column

data_bay_summary <- data_bay %>% 
  group_by(genetic_background, age, fly_id) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE), .names = "mean_{.col}"), 
            .groups = "drop") %>% 
  select(-mean_draw) %>% 
  ungroup() %>% 
  arrange(genetic_background, fly_id)

data_stats_summary <- data_stats_summary %>% 
  left_join(data_bay_summary) %>% 
  mutate(across(c(mean_k_star_emp, mean_k_star, mean_ttc), ~ log(. + 1), 
                .names = "{.col}_log")) %>%  
  logit_transform_monotonic(c("mean_p_ss_emp", "mean_p_reactivity_emp", "mean_m_abs",
                              "mean_m_rel", "mean_p_ss", "mean_p_0", "mean_p_reactivity")) %>% 
  mutate(age_factor = as.factor(age))

#Spontaneous jumping ---- 
data_accl <- data_raw %>% 
  filter(assay == "accl" & trial >240) %>% 
  group_by(exp_id, chamber) %>% 
  summarize(sum_spon_jumps = sum(jump == 1)) %>% 
  ungroup()
mean(data_accl$sum_spon_jumps)  
mean(data_accl$sum_spon_jumps) / 60 * 100 
sd(data_accl$sum_spon_jumps)  

# Average jump response curves----
data_ajr <- data_raw %>% 
  group_by(genetic_background, age, assay, trial) %>% 
  summarize(
    ajr = mean(jump, na.rm = TRUE),
    ) %>% 
  arrange(trial) 

##Graphing: Figure 1E and S7A----
subset_data <- subset(data_ajr, assay != "accl" & genetic_background == "KK") %>%
  mutate( 
    age = factor(age, levels = c(7, 14, 21)),
    assay = factor(assay, levels = unique(assay))) 
figure_1e <- ggplot(subset_data, aes(x = trial, y = ajr, color = assay)) +
  geom_line(linewidth = 0.5)  +
  labs(x = "Trial", 
       y = "Average jump response") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 100), breaks = c(50,100)) + 
  scale_y_continuous(expand = c(0, 0), limits = c(0, 0.7), breaks = seq(0, 1, by = 0.2)) +
  theme_minimal() +
  theme(
    text=element_text(family="sans", color ="black"),
    axis.line = element_line(linewidth = 0.5),
    axis.text = element_text(size = 8),
    axis.title.x = element_text(size = 10, margin = margin(t = 5)),
    axis.title.y = element_text(size = 10, margin = margin(r = 5)),
    panel.grid = element_line(colour = NULL),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = 'white', colour = 'white'),
    strip.background = element_rect(fill = "white", colour = "black"),
    panel.grid.major.x = element_line(color = "grey"),
    panel.grid.major.y = element_line(color = "grey"),
    legend.title = element_blank(),
    legend.key.size = unit(0.3, "cm"),
    legend.key.width= unit(0.75, "cm"),
    plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
    ) +
  facet_grid(. ~age)
figure_1e
ggsave(filename = "Figure 1E.pdf", plot = print(figure_1e), 
       width = 110, height = 65, units = "mm", dpi = 300)

subset_data <- subset(data_ajr, assay != "accl" & genetic_background == "GD") %>%
  mutate( 
    age = factor(age, levels = c(7, 14, 21)),
    assay = factor(assay, levels = unique(assay))) 
figure_s7a <- ggplot(subset_data, aes(x = trial, y = ajr, color = assay)) +
  geom_line(linewidth = 0.5)  +
  labs(x = "Trial", 
       y = "Average jump response") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 100), breaks = c(50,100)) + 
  scale_y_continuous(expand = c(0, 0), limits = c(0, 0.7), breaks = seq(0, 1, by = 0.2)) +
  theme_minimal() +
  theme(
    text=element_text(family="sans", color ="black"),
    axis.line = element_line(linewidth = 0.5),
    axis.text = element_text(size = 8),
    axis.title.x = element_text(size = 10, margin = margin(t = 5)),
    axis.title.y = element_text(size = 10, margin = margin(r = 5)),
    panel.grid = element_line(colour = NULL),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = 'white', colour = 'white'),
    strip.background = element_rect(fill = "white", colour = "black"),
    panel.grid.major.x = element_line(color = "grey"),
    panel.grid.major.y = element_line(color = "grey"),
    legend.title = element_blank(),
    legend.key.size = unit(0.3, "cm"),
    legend.key.width= unit(0.75, "cm"),
    plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
  ) +
  facet_grid(. ~age)
figure_s7a
ggsave(filename = "Figure S7A.pdf", plot = print(figure_s7a), 
       width = 110, height = 65, units = "mm", dpi = 300)

#Population-level jump rate in first 10 trials per hab assay ----
##Graphing: Figure S1A ----
figure_s1a <- ggplot(subset(data_stats, str_detect(assay,"hab") & genetic_background == "KK"), 
                    aes(y=p_transient_emp, x = assay)) + 
  geom_boxplot(outlier.size = 0.1, fill = "grey") + 
  scale_x_discrete(labels = 1:5) +
  labs(x = "First 10 stimuli per habituation assay", y = "Jump rate") + 
  theme(strip.background = element_rect(fill = "white", color = "black"), 
        panel.background = element_rect(fill = "white", color = "black"), 
        panel.grid.minor = element_blank(), 
        panel.grid.major.x = element_line(color = "grey", linewidth = 0.25), 
        panel.grid.major.y = element_line(color = "grey", linewidth = 0.25),
        text=element_text(family="sans", color ="black", size = 6)) +
  facet_grid(. ~age)
figure_s1a
ggsave(filename = "Figure S1A.pdf", plot = print(figure_s1a), 
       width = 80, height = 50, units = "mm", dpi = 300)

##Statistical analysis ----
lmer_s1a <- lmer(p_transient_emp_logit ~ (1|fly_id) + factor(age) + assay + chamber,
                data = subset(data_stats, str_detect(assay,"hab") & genetic_background == "KK"))
summary(lmer_s1a)
emm_age <- emmeans(lmer_s1a, ~ factor(age)) 
pairs(emm_age)
emm_assay <- emmeans(lmer_s1a, ~ assay) 
adjacent_contrasts <- contrast(emm_assay, method = "consec")
summary(adjacent_contrasts)


#Population-level jump rate per reactivity block ----
##Graphing: Figure S1B ----
figure_s1b <- ggplot(subset(data_sra_block, genetic_background == "KK"), 
                     aes(y=p_reactivity_block_emp, x = as.factor(block))) + 
  geom_boxplot(outlier.size = 0.1, fill = "grey") + 
  labs(x = "Reactivity block of 10 stimuli", y = "Jump rate") + 
  theme(strip.background = element_rect(fill = "white", color = "black"), 
        panel.background = element_rect(fill = "white", color = "black"), 
        panel.grid.minor = element_blank(), 
        panel.grid.major.x = element_line(color = "grey", linewidth = 0.25), 
        panel.grid.major.y = element_line(color = "grey", linewidth = 0.25),
        text=element_text(family="sans", color ="black", size = 6)) +
  facet_grid(. ~age)
figure_s1b
ggsave(filename = "Figure S1B.pdf", plot = print(figure_s1b), 
       width = 80, height = 50, units = "mm", dpi = 300)

##Statistical analysis ----
lmer_s1b <- lmer(p_reactivity_block_emp_logit ~ (1|fly_id) + age_factor + block_factor + chamber,
                 data = subset(data_stats_sra_block, genetic_background == "KK"))
summary(lmer_s1b)
emm_age <- emmeans(lmer_s1b, ~ age_factor)
pairs(emm_age)
emm <- emmeans(lmer_s1b, ~ block_factor) 
adjacent_contrasts <- contrast(emm, method = "consec")
summary(adjacent_contrasts)

#Mean Pairwise Squared Differences (MPSD) analyses ----
pairwise_squared_diff <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 2) return(NA)
  combs <- combn(x, 2)
  mean((combs[1, ] - combs[2, ])^2)
}

within_fly_psd <- data_stats %>%
  filter(str_detect(assay, "hab")) %>%
  group_by(genetic_background, age, fly_id) %>%
  summarise(across(c(p_ss_emp, m_abs_emp, m_rel_emp, k_star_emp),
                   ~ pairwise_squared_diff(.x),
                   .names = "psd_{.col}"),
            .groups = "drop") 

within_fly_psd_sra <- data_stats_sra_block %>%
  group_by(genetic_background, age, fly_id) %>%
  summarise(psd_p_reactivity_block_emp = pairwise_squared_diff(p_reactivity_block_emp),
            .groups = "drop")
within_fly_psd <- left_join(within_fly_psd, within_fly_psd_sra, 
                            by = c("genetic_background", "age", "fly_id"))

within_fly_psd_summary <- within_fly_psd %>% 
  group_by(genetic_background, age) %>%
  summarise(across(starts_with("psd_"),
                   ~ mean(.x, na.rm = TRUE),
                   .names = "{.col}_mean"),
            .groups = "drop") %>%
  pivot_longer(cols = starts_with("psd_"),
               names_to = "parameter",
               values_to = "within_psd") %>%
  mutate(parameter = str_remove(parameter, "^psd_"))


between_fly_psd <- data_stats %>%
  filter(str_detect(assay, "hab")) %>%
  group_by(genetic_background, age) %>%
  summarise(across(c(p_ss_emp, m_abs_emp, m_rel_emp, k_star_emp),
                   ~ {
                     df <- data.frame(
                       fly_id = pick(everything())$fly_id,
                       assay = pick(everything())$assay,
                       value = .x
                     ) %>%
                       filter(!is.na(value))
                     
                     mat <- df %>%
                       pivot_wider(names_from = assay, values_from = value, id_cols = fly_id) %>%
                       select(-fly_id) %>%
                       as.matrix()
                     
                     F <- nrow(mat)
                     T <- ncol(mat)
                     
                     if (F < 2 || T < 2) return(NA_real_)
                     
                     fly_pairs <- combn(F, 2)
                     time_pairs <- combn(T, 2)
                     sum_sq <- 0
                     
                     for (i in 1:ncol(fly_pairs)) {
                       f <- fly_pairs[1, i]
                       fp <- fly_pairs[2, i]
                       for (k in 1:ncol(time_pairs)) {
                         t <- time_pairs[1, k]
                         tp <- time_pairs[2, k]
                         if (!is.na(mat[f, t]) && !is.na(mat[fp, tp])) {
                           sum_sq <- sum_sq + (mat[f, t] - mat[fp, tp])^2
                         }
                       }
                     }
                     
                     denom <- choose(F, 2) * choose(T, 2)
                     sum_sq / denom
                   },
                   .names = "psd_{.col}"),
            .groups = "drop") %>%
  pivot_longer(cols = starts_with("psd_"),
               names_to = "parameter",
               values_to = "between_psd") %>%
  mutate(parameter = str_remove(parameter, "^psd_"))

between_fly_psd_sra <- data_stats_sra_block %>%
  group_by(genetic_background, age) %>%
  summarise(between_psd = {
    df <- data.frame(
      fly_id = pick(everything())$fly_id,
      block = pick(everything())$block,
      value = p_reactivity_block_emp
    ) %>%
      filter(!is.na(value))
    
    mat <- df %>%
      pivot_wider(names_from = block, values_from = value, id_cols = fly_id) %>%
      select(-fly_id) %>%
      as.matrix()
    
    F <- nrow(mat)
    T <- ncol(mat)
    
    if (F < 2 || T < 2) return(NA_real_)
    
    fly_pairs <- combn(F, 2)
    time_pairs <- combn(T, 2)
    sum_sq <- 0
    
    for (i in 1:ncol(fly_pairs)) {
      f <- fly_pairs[1, i]
      fp <- fly_pairs[2, i]
      for (k in 1:ncol(time_pairs)) {
        t <- time_pairs[1, k]
        tp <- time_pairs[2, k]
        if (!is.na(mat[f, t]) && !is.na(mat[fp, tp])) {
          sum_sq <- sum_sq + (mat[f, t] - mat[fp, tp])^2
        }
      }
    }
    
    denom <- choose(F, 2) * choose(T, 2)
    sum_sq / denom
  },
  .groups = "drop") %>%
  mutate(parameter = "p_reactivity_block_emp")

between_fly_psd <- bind_rows(between_fly_psd, between_fly_psd_sra)

##Graphing: Figure S6 and S8A ----
pairwise_squared_differences <- left_join(
  within_fly_psd_summary %>%
    mutate(parameter = str_remove(parameter, "_mean$"))
  , between_fly_psd)

pairwise_squared_differences <- pairwise_squared_differences %>% 
  mutate(parameter = factor(parameter, levels = c("p_reactivity_block_emp", "p_ss_emp",
                                                  "m_abs_emp", "m_rel_emp", "k_star_emp"),),
         genetic_background = factor(genetic_background, levels = c("KK", "GD"),),
         age = factor(age, levels = c("7", "14", "21"))
  )

for (lib in c("KK", "GD")) {
  df_sub <- pairwise_squared_differences %>% filter(genetic_background == lib)
  
  p <- ggplot(df_sub) +
    geom_line(
      aes(x = age, y = between_psd, group = 1, color = "Between flies"),
      linewidth = 1
    ) +
    geom_line(
      aes(x = age, y = within_psd,  group = 1, color = "Within flies"),
      linewidth = 1
    ) +
    facet_grid2(
      cols = vars(parameter),
      scales = "free_y",
      independent = "y",
      labeller = labeller(parameter = label_value)
    ) +
    coord_cartesian(ylim = c(0, NA)) +
    labs(x = "Age", y = "Pairwise Squared Difference") +
    scale_color_manual(
      values = c("Between flies" = "gray20", "Within flies" = "gray60"),
      name   = NULL
    ) +
    theme(
      strip.background   = element_rect(fill = "white", color = "black"),
      panel.background   = element_rect(fill = "white", color = "black"),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_line(color = "grey80"),
      panel.grid.major.y = element_line(color = "grey80"),
      text               = element_text(family = "sans", color = "black", size = 8),
      legend.position    = c(.9,.25)
    )
  assign(paste0("graph_psd_", lib), p)
}

graph_psd_KK
graph_psd_GD

ggsave(filename = "Figure S6.pdf", plot = print(graph_psd_KK), width = 170, height = 47, units = "mm", dpi = 300)
ggsave(filename = "Figure S8A.pdf", plot = print(graph_psd_GD), width = 170, height = 47, units = "mm", dpi = 300)


#Repeatability (R) analyses ----
##R within testing day----
###Run analyses ----
#Repeatability (consistency) within testing day for all empirical data statistics
parameters_emp_hab <- c("m_abs_emp", "m_rel_emp", "k_star_emp_log", "p_ss_emp_logit")

for (genetic_background1 in genetic_backgrounds) {
  for (parameter1 in parameters_emp_hab) { 
    for(age1 in ages) {
      subset_data <- subset(data_stats, str_detect(assay, "hab") & 
                              genetic_background == genetic_background1 & age == age1)
      formula <- as.formula(paste(parameter1, "~ (1|fly_id) + assay + chamber"))
      rep <- rpt(formula, grname = c("fly_id"), datatype = "Gaussian", data = subset_data, 
                 nboot = 1000, npermut = 1000)
      rep_name <- paste("rep", "age", genetic_background1, age1, parameter1,  sep = "")
      assign(rep_name, rep)
    }
  }
}

parameters_emp_sra <- c("p_reactivity_block_emp_logit")
for (genetic_background1 in genetic_backgrounds) {
  for (parameter1 in parameters_emp_sra) {
    for(age1 in ages) {
      subset_data <- subset(data_stats_sra_block, 
                            genetic_background == genetic_background1 & age == age1)
      formula <- as.formula(paste(parameter1, "~ (1|fly_id) + block_factor + chamber"))
      rep <- rpt(formula, grname = c("fly_id"), datatype = "Gaussian", data = subset_data, 
                 nboot = 1000, npermut = 1000)
      rep_name <- paste("rep", "age", genetic_background1, age1, parameter1,  sep = "")
      assign(rep_name, rep)
    }
  }
}

rep_within_age_objects <- ls(pattern = "^repage")
rep_within_age <- mget(rep_within_age_objects)

for (name in names(rep_within_age)) {
  data <- rep_within_age[[name]]
  cat(name)
  print(data)
  plot(data, type = "permut", cex.main = 1, sub = name)
}

###Creating csv with results ----
rep_within_age_results <- data.frame(      
  Name = character(),
  R = numeric(),
  SE = numeric(),
  CI_lower = numeric(),
  CI_upper = numeric(),
  P_LRT = numeric(),
  P_Permutation = numeric(),
  stringsAsFactors = FALSE
)
for (name in names(rep_within_age)) {
  data <- rep_within_age[[name]]
  if (all(c("R", "se", "CI_emp", "P") %in% names(data))) {
    rep_within_age_results <- rbind(rep_within_age_results, data.frame(            
      Name = name,
      R = data$R$fly_id,
      SE = data$se$se,
      CI_lower = data$CI_emp$`2.5%`,
      CI_upper = data$CI_emp$`97.5%`,
      P_LRT = data$P$LRT_P,
      P_Permutation = data$P$P_permut,
      stringsAsFactors = FALSE
    ))
  } else {
    cat("Skipping", name, "due to missing data\n")
  }
}

rep_within_age_results_KK <- rep_within_age_results %>%
  filter(str_detect(Name, "KK")) %>%
  arrange(desc(R)) %>% 
  mutate(p_perm_bh = p.adjust(P_Permutation, method = "BH"))

rep_within_age_results_GD <- rep_within_age_results %>%
  filter(str_detect(Name, "GD")) %>%
  arrange(desc(R)) %>% 
  mutate(p_perm_bh = p.adjust(P_Permutation, method = "BH"))

write.csv(rep_within_age_results_KK, "Repeatability within testing day KK.csv", row.names = FALSE)
write.csv(rep_within_age_results_GD, "Repeatability within testing day GD.csv", row.names = FALSE)


###Graphing: Figure 4C and S8B ----
parameters_levels <- c("p_reactivity_block_emp_logit", "p_ss_emp_logit", "m_abs_emp",
                       "m_rel_emp", "k_star_emp_log")
pattern <- paste(parameters_levels, collapse = "|")
custom_colors <- c("#d62728","#ff7f0e", "#2ca02c","#1f77b4", "#9467bd")

rep_within_age_for_graph <- rep_within_age_results %>% 
  filter(str_detect(Name, pattern)) %>% 
  mutate(Name = str_remove(Name, "^rep"),
         genetic_background = str_extract(Name, "(?<=age)[A-Z]{2}"),
         age = str_extract(Name, "(?<=age[A-Z]{2})\\d+"),
         age_factor = factor(age, levels = c("7", "14", "21")),
         parameter = str_remove(Name, "^age[A-Z]{2}\\d+"),
         parameter = factor(parameter, levels = parameters_levels)
  )

figure_4c <- ggplot(subset(rep_within_age_for_graph, genetic_background == "KK"), 
                    aes(x = age_factor, y = R, group = parameter)) +
  geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper, fill = parameter), alpha = 0.2) +  
  geom_line(aes(color = parameter), size = 1) +  
  geom_point(aes(fill = parameter), shape = 23) + 
  scale_color_manual(values = custom_colors) + 
  scale_fill_manual(values = custom_colors) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  facet_grid(~ parameter) +
  labs(x = "Age", y = "Repeatability (within day)") +
  theme(strip.background = element_rect(fill = "white", color = "black"), 
        panel.background = element_rect(fill = "white", color = "black"), 
        panel.grid.minor = element_blank(), 
        panel.grid.major.x = element_line(color = "grey"), 
        panel.grid.major.y = element_line(color = "grey"),
        text=element_text(family="sans", color ="black", size = 8), 
        legend.position = "none")
figure_4c
ggsave(filename = "Figure 4C.pdf", plot = print(figure_4c), 
       width = 170, height = 47, units = "mm", dpi = 300)

figure_s8b <- ggplot(subset(rep_within_age_for_graph, genetic_background == "GD"), 
                    aes(x = age_factor, y = R, group = parameter)) +
  geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper, fill = parameter), alpha = 0.2) +  
  geom_line(aes(color = parameter), size = 1) +  
  geom_point(aes(fill = parameter), shape = 23) + 
  scale_color_manual(values = custom_colors) + 
  scale_fill_manual(values = custom_colors) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  facet_grid(~ parameter) +
  labs(x = "Age", y = "R value") +
  theme(strip.background = element_rect(fill = "white", color = "black"), 
        panel.background = element_rect(fill = "white", color = "black"), 
        panel.grid.minor = element_blank(), 
        panel.grid.major.x = element_line(color = "grey"), 
        panel.grid.major.y = element_line(color = "grey"),
        text=element_text(family="sans", color ="black", size = 8), 
        legend.position = "none")
figure_s8b
ggsave(filename = "Figure S8B.pdf", plot = print(figure_s8b), 
       width = 170, height = 47, units = "mm", dpi = 300)

##R across age ----
###Run analyses ----
parameters_mean <- c("mean_p_0_logit", "mean_p_reactivity_logit", "mean_p_reactivity_emp_logit",
                     "mean_p_ss_logit", "mean_p_ss_emp_logit", "mean_m_abs_logit",
                     "mean_m_abs_emp", "mean_m_rel_logit", "mean_m_rel_emp", "mean_k_star_log",
                     "mean_k_star_emp_log")

for (genetic_background1 in genetic_backgrounds) {
  for (parameter1 in parameters_mean ) {
    formula <- as.formula(paste(parameter1, "~ (1|fly_id) + age_factor + chamber"))
    subset_data <- subset(data_stats_summary, genetic_background == genetic_background1)
    rep <- rpt(formula, grname = c("fly_id"), datatype = "Gaussian", data = subset_data, 
               nboot = 1000, npermut = 1000)
    rep_name <- paste("rep_across_age_", genetic_background1, parameter1, sep = "")
    assign(rep_name, rep)
  }
}
rep_across_age_objects <- ls(pattern = "^rep_across_age_")
rep_across_age <- mget(rep_across_age_objects)

for (name in names(rep_across_age)) {
  data <- rep_across_age[[name]]
  cat(name)
  print(data)
  plot(data, type = "permut", cex.main = 1, sub = name)
}

###Creating csv with results ----
rep_across_age_results <- data.frame(
  Name = character(),
  R = numeric(),
  SE = numeric(),
  CI_lower = numeric(),
  CI_upper = numeric(),
  P_LRT = numeric(),
  P_Permutation = numeric(),
  stringsAsFactors = FALSE
)
for (name in names(rep_across_age)) {
  data <- rep_across_age[[name]]
  if (all(c("R", "se", "CI_emp", "P") %in% names(data))) {
    rep_across_age_results <- rbind(rep_across_age_results, data.frame(
      Name = name,
      R = data$R$fly_id,
      SE = data$se$se,
      CI_lower = data$CI_emp$`2.5%`,
      CI_upper = data$CI_emp$`97.5%`,
      P_LRT = data$P$LRT_P,
      P_Permutation = data$P$P_permut,
      stringsAsFactors = FALSE
    ))
  } else {
    cat("Skipping", name, "due to missing data\n")
  }
}

rep_across_age_resultss_KK <- rep_across_age_results %>%
  filter(str_detect(Name, "KK")) %>%
  arrange(desc(R)) %>% 
  mutate(p_perm_bh = p.adjust(P_Permutation, method = "BH"))

rep_across_age_results_GD <- rep_across_age_results %>%
  filter(str_detect(Name, "GD")) %>%
  arrange(desc(R)) %>% 
  mutate(p_perm_bh = p.adjust(P_Permutation, method = "BH"))

write.csv(rep_across_age_resultss_KK, "Repeatability across age KK.csv", row.names = FALSE)
write.csv(rep_across_age_results_GD, "Repeatability across age GD.csv", row.names = FALSE)


###Graphing: Figure 4D and S8C ----
parameters_mean_levels <- c("mean_p_0_logit", "mean_p_reactivity_logit",
                            "mean_p_reactivity_emp_logit", "mean_p_ss_logit",
                            "mean_p_ss_emp_logit", "mean_m_abs_logit", "mean_m_abs_emp",
                            "mean_m_rel_logit", "mean_m_rel_emp", "mean_k_star_log",
                            "mean_k_star_emp_log")

pattern <- paste(parameters_mean_levels, collapse = "|")

rep_across_age_for_graph <- rep_across_age_results %>% 
  filter(str_detect(Name, pattern)) %>% 
  mutate(Name = str_remove(Name, "^rep_across_age_"),
         genetic_background = str_extract(Name, "^[A-Z]{2}"),
         parameter = str_remove(Name, "^[A-Z]{2}"),
         parameter = factor(parameter, levels = parameters_mean_levels))

custom_colors2 <- rep(custom_colors,each =2)
custom_shapes <- rep(c(21,23), 5)
figure_4d <- ggplot(subset(rep_across_age_for_graph, genetic_background == "KK"), 
                    aes(x = parameter, y = R, color = parameter)) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.3) +
  geom_point(stroke = 0.5, color = "black", size = 2, 
             aes(fill = parameter, shape = parameter)) +
  scale_shape_manual(values = c(21, custom_shapes)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  scale_fill_manual(values = c("darkred", custom_colors2)) +  
  scale_color_manual(values = c("darkred",custom_colors2)) + 
  labs(x = " ", y = "Repeatability (across ages)") +
  theme(strip.background = element_rect(fill = "white", color = "black"), 
        panel.background = element_rect(fill = "white", color = "black"), 
        panel.grid.minor = element_blank(), 
        panel.grid.major.x = element_line(color = "grey"), 
        panel.grid.major.y = element_line(color = "grey"), 
        axis.text.x = element_text(angle = 45, hjust=1), 
        text=element_text(family="sans", color ="black", size = 8), 
        legend.position = "none")
figure_4d
ggsave(filename = "Figure 4D.pdf", plot = print(figure_4d), 
       width = 170, height = 63, units = "mm", dpi = 300)


figure_s8c <- ggplot(subset(rep_across_age_for_graph, genetic_background == "GD"), 
                    aes(x = parameter, y = R, color = parameter)) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.3) +
  geom_point(stroke = 0.5, color = "black", size = 2, 
             aes(fill = parameter, shape = parameter)) +
  scale_shape_manual(values = c(21, custom_shapes)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  scale_fill_manual(values = c("darkred", custom_colors2)) +  
  scale_color_manual(values = c("darkred",custom_colors2)) + 
  labs(x = " ", y = "Repeatability") +
  theme(strip.background = element_rect(fill = "white", color = "black"), 
        panel.background = element_rect(fill = "white", color = "black"), 
        panel.grid.minor = element_blank(), 
        panel.grid.major.x = element_line(color = "grey"), 
        panel.grid.major.y = element_line(color = "grey"), 
        axis.text.x = element_text(angle = 45, hjust=1), 
        text=element_text(family="sans", color ="black", size = 8), 
        legend.position = "none")
figure_s8c
ggsave(filename = "Figure S8C.pdf", plot = print(figure_s8c), 
       width = 170, height = 63, units = "mm", dpi = 300)

#Histograms of SD across age ----
data_stats_summary_sd <- data_stats_summary %>%
  select(-matches("log")) %>%
  group_by(genetic_background, fly_id) %>%
  summarise(across(starts_with("mean_"), 
                   ~ {
                     if (sum(!is.na(.x)) > 1) {
                       sd(.x, na.rm = TRUE)
                     } else {
                       NA_real_
                     }
                   },
                   .names = "sd_{.col}")
            ) %>%
  ungroup()

## Graphing: Figure 4I ----
parameters_sd_mean <- c("sd_mean_p_0", "sd_mean_p_ss") 
pdf("Figure 4I.pdf")
for (param in parameters_sd_mean) {
  values <- subset(data_stats_summary_sd, genetic_background == "KK")[[param]]
  values <- na.omit(values)
  hist(values, breaks = 30, main = param)
}
dev.off()
