library(tableone)
dbfilename<-paste0("/Users/wangyirui/Downloads", "/predictions.txt")
db_analysis<-read.table(dbfilename, header = FALSE, sep = ",", 
                               col.names = c("file_name", "label", "prediction", "ene_label", "ene_prediction"))


####ROC plot####
#pROC
library(pROC)
library(ggplot2)
library(dplyr)
obj<-roc(db_analysis$label~db_analysis$prediction)

  ciobj <- ci.se(obj, specificities = seq(0, 1, l = 100))
  dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                       lower = ciobj[, 1],
                       upper = ciobj[, 3])
  youden_j<-data.frame(as.list(coords(obj, x=0.15, input="threshold"))) %>%
    mutate(inv_spec=1-specificity,
           youden_j=sensitivity+specificity-1);youden_j
  
  
  ggroc(obj,legacy.axes = TRUE) +
#    theme_minimal() +
    theme_linedraw() +
    labs(x = "1 - Specificity", y = "Sensitivity", linetype="Gonial Angle Measurement") +
   scale_y_continuous(breaks = seq(0,1,0.05),limits=c(0,1))+
   scale_x_continuous(breaks = seq(0,1,0.05),limits = c(0,1))+
    # scale_y_continuous(breaks = seq(0,1,0.05),limits=c(0.7,1))+
    # scale_x_continuous(breaks = seq(0,1,0.05),limits = c(0,0.3))+
    
    scale_linetype_discrete(labels=c("Mean Gonial","Gonial Angle Left","Gonial Angle Right"))+
    geom_point(data=youden_j, aes(inv_spec, sensitivity), shape=3, size=3, fill="red") +
    ggtitle("ROC curve") +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      alpha = 0.7,
      color = "grey"
    ) + coord_equal() +
    geom_ribbon(
      data = dat.ci,
      aes(x = 1-x, ymin = lower, ymax = upper),
      fill = "blue",
      alpha = 0.2
    ) 
ci.auc(obj)


####Plot PR curve####

PRdata<-coords(obj, x=seq(0,1,l=100), input="sensitivity",ret = c("recall", "precision"), transpose = FALSE)
ciPPVobj <- ci.coords(obj, x = seq(0, 1, l = 100),input="sensitivity",ret="ppv")
dat.PPVci <- data.frame(x = as.numeric(attributes(ciPPVobj)$x),
                     lower = ciPPVobj$ppv[, 1],
                     upper = ciPPVobj$ppv[, 3])
youden_j_PR<-data.frame(as.list(coords(obj, x=0.15, input="threshold",ret=c("sensitivity","ppv"))))


ggplot(PRdata) +
  geom_line(aes(x=recall,y=precision))+
  geom_hline(yintercept = PRdata$precision[which(PRdata$recall==1)], colour = "grey",
             linetype = 3)+
  geom_point(data=youden_j_PR, aes(sensitivity, ppv), shape=3, size=3, fill="red") +
  theme_minimal() +
#  theme_linedraw() +
  labs(x = "Sensitivity", y = "Positive predict value", linetype="Gonial Angle Measurement") +
 scale_y_continuous(breaks = seq(0,1,0.1),limits=c(0,1))+
 scale_x_continuous(breaks = seq(0,1,0.1),limits = c(0,1))+
  # scale_y_continuous(breaks = seq(0,1,0.05),limits=c(0.7,1) )+
  # scale_x_continuous(breaks = seq(0,1,0.05),limits = c(0.7,1))+
  scale_linetype_discrete(labels=c("Mean Gonial","Gonial Angle Left","Gonial Angle Right")) +
#  geom_point(data=youden_j, aes(inv_spec, sensitivity), shape=3, size=3, fill="red") +
  ggtitle("PR curve") +
  coord_equal() +
  geom_ribbon(
    data = dat.PPVci,
    aes(x = x, ymin = lower, ymax = upper),
    fill = "blue",
    alpha = 0.2
  ) 

library(precrec)

# Generate an mscurve object that contains ROC and Precision-Recall curves
mmcurves <- evalmod(scores = db_analysis$prediction, labels =  db_analysis$label)
PRAUC<-auc(mmcurves)['aucs'][2,]

####Calculate CI for AUROC and AUPRC####
ciobj <- ci.se(obj, specificities = seq(0, 1, l = 1000))
dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                     lower = ciobj[, 1],
                     upper = ciobj[, 3])

AUROC.lower<-sum(dat.ci$lower)/1000
AUROC.upper<-sum(dat.ci$upper)/1000

ciPPVobj <- ci.coords(obj, x = seq(0, 1, l = 1000),input="sensitivity",ret="ppv")
dat.PPVci <- data.frame(x = as.numeric(attributes(ciPPVobj)$x),
                        lower = ciPPVobj$ppv[, 1],
                        upper = ciPPVobj$ppv[, 3])
AUPRC.lower<-sum(dat.PPVci$lower,na.rm=TRUE)/1000
AUPRC.upper<-sum(dat.PPVci$upper,na.rm=TRUE)/1000

###  
  
ci.auc(obj)
  
  
  
  
  





########
ROC_calculation<-function(db_ROC,subset_name){
  #db_ROC<-db_testset
  #subset_name<-"all cases"
  cutoff<-0.10
  rets <- c("threshold","specificity", "sensitivity", "accuracy", "tn", "tp", "fn", "fp", "npv", 
            "ppv", "1-specificity", "1-sensitivity", "1-accuracy", "1-npv", "1-ppv")
  ROC_overall<-roc(db_ROC$Abnormal~db_ROC$prob)
  ROC_clinical<-roc(db_ROC$clinical_sig~db_ROC$prob)
  #S95_overall<-ci.coords(ROC_overall,x=0.95,input="sensitivity",ret=c("specificity","accuracy"))
  S95_overall<-coords(ROC_overall,x=0.95,input="sensitivity",ret=rets,transpose=FALSE)
  S95_clinical<-coords(ROC_clinical,x=0.95,input="sensitivity",ret=rets,transpose=FALSE)
  cutoff_overall<-coords(ROC_overall,x=cutoff,input="threshold",ret=rets,transpose=FALSE)
  cutoff_clinical<-coords(ROC_clinical,x=cutoff,input="threshold",ret=rets,transpose=FALSE)
  best_overall<-coords(ROC_overall,"best",ret=rets,transpose=FALSE)
  best_clinical<-coords(ROC_clinical,"best",ret=rets,transpose=FALSE)
  AUC_overall<-ci.auc(ROC_overall)
  AUC_clinical<-ci.auc(ROC_clinical)
  
  S95_overall$title<-"S95 overall"
  S95_clinical$title<-"S95 clinical"
  cutoff_overall$title<-"cutoff overall"
  cutoff_clinical$title<-"cutoff clinical"
  best_overall$title<-"best overall"
  best_clinical$title<-"best clinical"
  
  
  result_tb<-rbind(S95_overall,S95_clinical,best_overall,best_clinical,cutoff_overall,cutoff_clinical)
  result_tb$dataset<-subset_name
  results<-list(result_tb,ROC_overall,ROC_clinical)
  return(results)
}
allcase_result<-ROC_calculation(db_testset,"all cases")
hip_result<-ROC_calculation(subset(db_testset,db_testset$Hipfx==1 | (db_testset$Hipfx==0 & (db_testset$Pelvicfx!=1 & db_testset$FShaftfx!=1 & db_testset$Dislocation!=1 & db_testset$Periprothesis!=1 & db_testset$Other!=1))),"Hip")
hip_dislocation_result<-ROC_calculation(subset(db_testset,db_testset$Hipfx==1 | db_testset$Dislocation!=1 | (db_testset$Hipfx==0 & (db_testset$Pelvicfx!=1 & db_testset$FShaftfx!=1 & db_testset$Periprothesis!=1 & db_testset$Other!=1))),"Hip_dislocate")
pelvic_result<-ROC_calculation(subset(db_testset,db_testset$Pelvicfx==1 | (db_testset$Pelvicfx==0 & (db_testset$Hipfx!=1 & db_testset$FShaftfx!=1 & db_testset$Dislocation!=1 & db_testset$Periprothesis!=1 & db_testset$Other!=1))),"pelvic")
hip_pelvic_result<-ROC_calculation(subset(db_testset, db_testset$Hipfx==1 | db_testset$Pelvicfx==1 | (db_testset$Hipfx==0 & db_testset$Pelvicfx==0 & (db_testset$FShaftfx!=1 & db_testset$Dislocation!=1 & db_testset$Periprothesis!=1 & db_testset$Other!=1))),"Hip pelvic")

all_results<-rbind(as.data.frame(allcase_result[1]),as.data.frame(hip_result[1]),as.data.frame(pelvic_result[1]),as.data.frame(hip_pelvic_result[1]))

if (writefile==TRUE){
  write.csv(all_results,paste0(resultpath,"all_results.csv"))
}

ROCobj<-roc(db_testset$Abnormal~db_testset$prob)
ci.auc(ROCobj)

db_testset %>%
  ggplot(aes(x=prob,fill=Abnormal)) + 
  geom_density(alpha = 0.2) +
  scale_x_continuous(breaks=seq(0,1,0.05))
#    geom_histogram(binwidth=0.05)

difficult_set<-subset(db_testset, Abnormal=="TRUE" & Difficult.Case=="TRUE")
length(which(difficult_set$prob>0.15))/length(difficult_set$EDID)
clnicalsig_set<-subset(db_testset, Abnormal=="TRUE" & clinical_sig==1)
length(which(clnicalsig_set$prob>0.15))/length(clnicalsig_set$EDID)
table(db_testset$Abnormal)


####annotation numbers####
trainset<-subset(db_analysis, datasplit=='train')
summary(trainset$annotation_nubmer)
sum(trainset$annotation_nubmer,na.rm=T)
table(trainset$hipfracture)
table(trainset$Abnormal)
sum(trainset$hipfx_intracapsular+trainset$hipfx_extracapsular==2)

####test set####
testset<-subset(db_analysis, datasplit=='testing')
table(testset$Abnormal)
table(testset[which(testset$predict_category=='FN'),'Hipfx'])
table(testset[which(testset$predict_category=='FN'),'Hipfx'])/sum(testset$predict_category=='FN')
table(testset[which(testset$predict_category=='FN'),'Pelvicfx'])
table(testset[which(testset$predict_category=='FN'),'Pelvicfx'])/sum(testset$predict_category=='FN')
table(testset[which(testset$predict_category=='FN'),'Dislocation'])
table(testset[which(testset$predict_category=='FN'),'Dislocation'])/sum(testset$predict_category=='FN')
table(testset[which(testset$predict_category=='FN'),'Periprothesis'])
table(testset[which(testset$predict_category=='FN'),'Periprothesis'])/sum(testset$predict_category=='FN')
table(testset[which(testset$predict_category=='FN'),'FShaftfx'])
table(testset[which(testset$predict_category=='FN'),'FShaftfx'])/sum(testset$predict_category=='FN')

table(testset$FShaftfx,testset$predict_category)
table(testset$Hipfx,testset$predict_category)
table(testset$Pelvicfx,testset$predict_category)
table(testset$Dislocation,testset$predict_category)
table(testset$Periprothesis,testset$predict_category)

tablevars<-c("predict_category","Hipfx","Pelvicfx","FShaftfx","Dislocation","Periprothesis")
supptb1<-CreateTableOne(vars = tablevars,strata = 'predict_category',data = subset(testset,predict_category=='FN'|predict_category=='TP'), factorVars = tablevars)
write.csv(print(supptb1, showAllLevels = TRUE,missing =TRUE,noSpaces = TRUE, exact = tablevars,format = "f"),paste0(resultpath,"Table_supp1.csv"))

