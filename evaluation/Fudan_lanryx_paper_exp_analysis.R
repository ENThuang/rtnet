library(tableone)
library(pROC)
library(ggplot2)
library(dplyr)

exp_root <- "/nas/yirui.wang/exps/Fudan_HN_LN_paper/final_results"
subpath <- "/last_epoch/mobilenetv3_large_25d_fudan_hn_ln_bce_loss_dual_maxpool_ENE/predictions.txt"

# used to calculate the mean of ensembled five-fold cross-validation results
internal_meta_run_ensemble <- list()
internal_ene_run_ensemble <- list()
# used to calculate the mean of 5 * 5 ensembled results
external_meta_run_ensemble <- list()
external_ene_run_ensemble <- list()

col_fname <- "fname"
col_meta_label <- "meta_label"
col_meta_pred <- "meta_pred"
col_ene_label <- "ene_label"
col_ene_pred <- "ene_pred"

external_centers <- list("FudanENT", "CGMHLarynx", "CGMGOral", "TCGA")

# Loop through the fold numbers
for (run_idx in 0:4) {
    for (fold_idx in 0:4) {
        file_name <- paste0(exp_root, "RUN", run_idx, "/fold", fold_idx, subpath)
        fold_results <- read.table(dbfilename, 
                                header = FALSE, 
                                sep = ",", 
                                col.names = c(col_fname, col_meta_label, col_meta_pred, col_ene_label, col_ene_pred))
        # select internal testing results (not containing "ext" in the file name)
        int_meta_label <- fold_results&col_meta_label[!grepl("ext", fold_results$col_fname), ]
        int_meta_pred <- fold_results&col_meta_pred[!grepl("ext", fold_results$col_fname), ]
        

        int_ene_label <- fold_results&col_ene_label[!grepl("ext", fold_results$col_fname), ]
        int_ene_pred <- fold_results&col_ene_pred[!grepl("ext", fold_results$col_fname), ]


        # Store the data in the list
        fold_results[[i]] <- fold_data
    }
  
}

