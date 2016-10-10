rm(list = ls())

setwd("/home/fractaluser/Documents/DMA-Challenge")

library(data.table)
library(caret)
library(Matrix)
library(xgboost)
library(corrgram)
library(ggplot2)
library(ROCR)
library(pROC)
library(lubridate)
library(rpart)

train <- fread("EY_DMA_Analytics_2016_Training_Data0822.csv", stringsAsFactors = FALSE)

##### Capping profitability at 1st and 99th percentile values

quantile(train$Profitability, prob = c(0.01, 0.99))

train$Profitability[train$Profitability < -31200] <- -31200
train$Profitability[train$Profitability > 5800] <- 5800

train$total_rev_hi_lim <- log(train$total_rev_hi_lim + 1)
train$total_rev_hi_lim[is.na(train$total_rev_hi_lim)] <- mean(train$total_rev_hi_lim, na.rm = TRUE)
train$total_rev_hi_lim_outlierflag <- ifelse(train$total_rev_hi_lim > 12, 1, 0)
train$total_rev_hi_lim[train$total_rev_hi_lim > 12] <- 12

train$tot_cur_bal <- log(train$tot_cur_bal+1)
train$tot_cur_bal[is.na(train$tot_cur_bal)] <- mean(train$tot_cur_bal, na.rm = TRUE)
train$tot_cur_bal[train$tot_cur_bal > 13.5] <- 13.5

train$annual_inc <- log(train$annual_inc + 1)
train$annual_inc[train$annual_inc > 13] <- 13


train$purpose <- factor(train$purpose)
train$emp_length <- factor(train$emp_length, levels = c("n/a","< 1 year","1 year",
                                                        "2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"))

# train$emp_length_lessthan1year <- ifelse(train$emp_length=="< 1 year", 1, 0)
# train$emp_length_lessthan4year <- ifelse(train$emp_length %in% c("< 1 year", "1 year","2 years","3 years","4 years"), 1, 0)
# train$emp_length_lessthan7year <- ifelse(train$emp_length %in% c("< 1 year", "1 year","2 years","3 years","4 years",
#                                                                  "5 years","6 years","7 years"), 1, 0)

train$term <- factor(train$term, levels = c(" 36 months", " 60 months"))
train$home_ownership[train$home_ownership %in% c("NONE", "OTHER", "ANY")] <- "NONE+OTHER+ANY"
train$home_ownership <- factor(train$home_ownership, levels = c("NONE+OTHER+ANY","MORTGAGE", "RENT", "OWN"))

train$verification_status <- factor(train$verification_status, levels = c("Not Confirmed","Confirmed", "Source Confirmed"))

train$mths_since_last_delinq[is.na(train$mths_since_last_delinq)] <- median(train$mths_since_last_delinq, na.rm = TRUE)
train$initial_list_status <- factor(train$initial_list_status)


# train$issue_month_factor <- factor(substring(train$issue_d, 1, 3), levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
#                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

train$issue_d <- paste0("01-", train$issue_d)
train$earliest_cr_line <- paste0("01-", train$earliest_cr_line)

train$issue_d <- dmy(train$issue_d)
train$earliest_cr_line <- dmy(train$earliest_cr_line)

train$partial_age <- as.numeric(train$issue_d - train$earliest_cr_line)/365
train$partial_age[is.na(train$partial_age)] <- median(train$partial_age, na.rm = TRUE)
train$partial_age[train$partial_age > 40] <- 40

train$collections_12_mths_ex_med[is.na(train$collections_12_mths_ex_med)] <- 0
train$collections_12_mths_ex_med[train$collections_12_mths_ex_med > 2] <- 2

train$dti[train$dti > 40] <- 40

train$open_acc[train$open_acc > 30] <- 30

train$pub_rec[train$pub_rec > 2] <- 2
train$pub_rec_flag <- ifelse(train$pub_rec >= 1, 1, 0)

train$revol_bal <- log(train$revol_bal + 1)
train$revol_bal[train$revol_bal > 11.5] <- 11.5

train$total_rec_late_fee_flag <- ifelse(train$total_rec_late_fee > 0, 1, 0)

train$desc_flag <- ifelse(train$desc=="", 1, 0)

train$issue_month <- month(train$issue_d)
train$issue_year <- year(train$issue_d)

# train$last_pymnt_d <- paste0("01-", train$last_pymnt_d)
# train$last_pymnt_d <- dmy(train$last_pymnt_d)
# 
# train$last_payment_month <- month(train$last_pymnt_d)
# train$last_pymnt_year <- year(train$last_pymnt_d)

train$last_credit_pull_d <- paste0("01-", train$last_credit_pull_d)
train$last_credit_pull_d <- dmy(train$last_credit_pull_d)

train$last_credit_pull_month <- month(train$last_credit_pull_d)
train$last_credit_pull_year <- year(train$last_credit_pull_d)

train$earliest_cr_line_month <- month(train$earliest_cr_line)
train$earliest_cr_line_year <- year(train$earliest_cr_line)

# test$earliest_cr_line_month <- month(test$earliest_cr_line)
# test$earliest_cr_line_year <- year(test$earliest_cr_line)

#replaced missing values of years with 2016 and month with 1 based on frequency

train$last_credit_pull_year[is.na(train$last_credit_pull_year)] <- 2016
train$last_credit_pull_month[is.na(train$last_credit_pull_month)] <- 1


## last payment month

train$last_pymnt_d <- paste0("01-", train$last_pymnt_d)
train$last_pymnt_d <- dmy(train$last_pymnt_d)

train$last_pymnt_month <- month(train$last_pymnt_d)
train$last_pymnt_year <- year(train$last_pymnt_d)

train$last_pymnt_year[is.na(train$last_pymnt_year)] <- 2016
train$last_pymnt_month[is.na(train$last_pymnt_month)] <- 1


# train$next_pymnt_d <- paste0("01-",train$next_pymnt_d)
# train$next_pymnt_d <- dmy(train$next_pymnt_d)
# train$next_pymnt_month <- month(train$next_pymnt_d)
# 
# train$next_pymnt_month[is.na(train$next_pymnt_month)] <- 2

####Text mining results

train$consolidation_flag <- 0
train$consolidation_flag[grep("consolid", train$title)] <- 1


#### Month columns############################################################

train$issue_month_flag <- ifelse(train$issue_month==12, 1, 0)
train$last_pymnt_month_flag <- ifelse(train$last_pymnt_month %in% c(1,12), 1, 0)
train$last_credit_pull_month_flag <- ifelse(train$last_credit_pull_month==1, 1, 0)

############################################################

  
  
#### Logistic model for predicting loan_status

drop <- c("issue_d", "last_pymnt_d", "last_credit_pull_d", "id", "member_id", 
          "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
          "annual_inc_joint", "desc", "title", "zip_code", "addr_state", 
          "verification_status_joint", "earliest_cr_line", 
          
          "funded_amnt", "funded_amnt_inv", "loan_amnt", "out_prncp", "out_prncp_inv", "emp_length", 
          "Profitability", "application_type", "home_ownership", "revol_util", "acc_now_delinq",
          "initial_list_status", "collections_12_mths_ex_med", "total_rev_hi_lim_outlierflag",
          "desc_flag", "last_credit_pull_year", "total_rev_hi_lim", "emp_length_lessthan1year")

train.all <- train
train <- train[,which(!names(train) %in% drop), with = FALSE]

train$loan_status <- ifelse(train$loan_status %in% c("Charge Off", "Current"), "Bad", "Good")
train$loan_status <- factor(train$loan_status)

# set.seed(10)

small <- createDataPartition(train$loan_status, times = 1, p = 0.1, list = FALSE)

fit1 <- glm(loan_status ~ ., data = train[small], family = "binomial")

summary(fit1)

train2 <- train[-small]
train2 <- train2[sample(1:nrow(train2), 10000)]

pred1 <- predict(fit1, train2, type = "response")

plot(performance(prediction(pred1, train2$loan_status), 'acc'))

for(i in seq(0.2, 0.5, 0.01)){

  class1 <- ifelse(pred1 > i, "Good", "Bad")

  print(paste0(i, " - ", confusionMatrix(class1, train2$loan_status)$overall[["Accuracy"]]))

}

### 0.33 seems to be right cut-off value with maximum accuracy

train.all$loan_status_prob <- predict(fit1, train.all, type = "response")
train.all$loan_status_pred <- ifelse(train.all$loan_status_prob >=0.33, "Good", "Bad") 
train.all$loan_status_pred <- factor(train.all$loan_status_pred, levels = c("Bad", "Good"))

train.all$score <- (train.all$loan_status_prob * train.all$dti)*100
###### Linear regression model for predicting profitability as input to boosting trees


drop <- c("issue_d", "last_pymnt_d", "last_credit_pull_d", "id", "member_id", 
          "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
          "annual_inc_joint", "desc", "title", "zip_code", "addr_state", 
          "verification_status_joint", "earliest_cr_line", 
          
          "funded_amnt", "funded_amnt_inv", "loan_amnt", "out_prncp", "out_prncp_inv", 
          "home_ownership", "acc_now_delinq", "revol_util", "loan_status", "revol_bal", 
          "total_rec_late_fee_flag", "total_rev_hi_lim_outlierflag", "pub_rec_flag", "total_acc",
          "tot_cur_bal", "emp_length", "collections_12_mths_ex_med", "desc_flag", "delinq_2yrs")

train <- train.all[,which(!names(train.all) %in% drop), with = FALSE]

# set.seed(11)

small <- createDataPartition(train$Profitability, times = 1, p = 0.05, list = FALSE)

train2 <- train[-small]
train2 <- train2[sample(1:nrow(train2), 10000)]

#linear model
fit2 <- lm(Profitability ~ ., data = train[small])

pred2 <- predict(fit2, train2)

RMSE(pred2, train2$Profitability)


# #boosted tree model
# 
# #1
# train.matrix <- sparse.model.matrix(Profitability ~. , data = train[small])
# test.matrix <- sparse.model.matrix(Profitability ~., data = train2)
# 
# fit3 <- xgboost(data = train.matrix ,label = train[small]$Profitability, max.depth = 10,
#               nround = 50,objective = "reg:linear")
# 
# pred3 <- predict(fit3, test.matrix)
# 
# RMSE(pred3, train2$Profitability)
# 
# #2
# ctrl <- trainControl(allowParallel = TRUE)
# 
# fit4 <- train(Profitability ~ ., data = train[small], method = "gbm", verbose = FALSE, trControl = ctrl)
# 
# pred4 <- predict(fit4, test2)
# 
# RMSE(pred4, train2$Profitability)

# single tree  model

fit4 <- rpart(Profitability ~ ., data = train[small])

pred4 <- predict(fit4, train2)

RMSE(pred4, train2$Profitability)

#### Checking multicollinearity

# numcols <- names(train.all)[sapply(train.all, function(x) is.numeric(x))]
# 
# train2.num <- train2[,which(names(train2) %in% numcols), with = FALSE]
# 
# corrgram(train2.num, lower.panel = panel.pie)
# names(train2)

##################################
train.all$lm_pred <- predict(fit2, train.all)
train.all$lm_pred_flag <- 1
train.all$lm_pred_flag[train.all$Profitability < 0] <- -1

train.all$tree_pred <- predict(fit4, train.all) 

train.all$purpose_flag1 <- ifelse(train.all$purpose %in% c("educational", "wedding"), 1, 0)
train.all$purpose_flag2 <- ifelse(train.all$purpose %in% c("small_business"), 1, 0)

train.all$verification_status_flag <- ifelse(train.all$verification_status=="Not Confirmed", 1, 0)
train.all$verification_status_joint_flag <- ifelse(train.all$verification_status_joint %in% c("Source Verified", "Verified"), 1, 0)


################# trying boosting 

# set.seed(12)

train.all$state_flag1 <- ifelse(train.all$addr_state %in% c("MS", "ME", "ND", "NE"), 1, 0) 
train.all$state_flag2 <- ifelse(train.all$addr_state %in% c("ID","IA"), 1, 0) 


inVal <- createDataPartition(train.all$Profitability, times = 1, p = 0.2, list = FALSE)

validation <- train.all[inVal]
train.all <- train.all[-inVal]
rm(inVal)

inTrain <- createDataPartition(train.all$Profitability, times = 1, p = 0.5, list = FALSE)

train <- data.frame(train.all[inTrain])
test <- data.frame(train.all[-inTrain])


train.small <- train[, c("Profitability", "loan_status_pred", "lm_pred", "tree_pred","loan_amnt", "funded_amnt", "funded_amnt_inv", "term",
                        "total_rec_late_fee", "state_flag1", "lm_pred_flag","purpose","purpose_flag1", "purpose_flag2",
                        "verification_status", "partial_age", "annual_inc", "issue_month", "issue_year",
                        "last_credit_pull_month", "last_credit_pull_year","mths_since_last_delinq",
                        "last_pymnt_month", "last_pymnt_year", "score", 
                        "issue_month_flag", "last_pymnt_month_flag","last_credit_pull_month_flag")] 

test.small <- test[, c("Profitability", "loan_status_pred", "lm_pred", "tree_pred","loan_amnt", "funded_amnt", "funded_amnt_inv", "term",
                        "total_rec_late_fee", "state_flag1", "lm_pred_flag","purpose","purpose_flag1", "purpose_flag2",
                        "verification_status", "partial_age", "annual_inc", "issue_month", "issue_year",
                        "last_credit_pull_month", "last_credit_pull_year","mths_since_last_delinq",
                       "last_pymnt_month", "last_pymnt_year", "score", 
                       "issue_month_flag", "last_pymnt_month_flag","last_credit_pull_month_flag")] 


train.matrix <- sparse.model.matrix(Profitability ~ ., data = train.small) 
test.matrix <- sparse.model.matrix(Profitability ~ ., data = test.small) 

boostfit1  <- xgboost(data = train.matrix,label = train.small$Profitability, max.depth = 10,
                nthread = 4, nround = 20,objective = "reg:linear", lambda = 5)

RMSE(predict(boostfit1, test.matrix), test.small$Profitability)


#####################################################################
###################final check on validation set##############################
#####################################################################


validation <- data.frame(validation)

validation.small <- validation[, c("Profitability", "loan_status_pred", "lm_pred", "tree_pred","loan_amnt", "funded_amnt", "funded_amnt_inv", "term",
                                   "total_rec_late_fee", "state_flag1", "lm_pred_flag","purpose","purpose_flag1", "purpose_flag2",
                                   "verification_status", "partial_age", "annual_inc", "issue_month", "issue_year",
                                   "last_credit_pull_month", "last_credit_pull_year","mths_since_last_delinq",
                                   "last_pymnt_month", "last_pymnt_year", "score", 
                                   "issue_month_flag", "last_pymnt_month_flag","last_credit_pull_month_flag")] 

valid.matrix <- sparse.model.matrix(Profitability ~., data = validation.small)
RMSE(predict(boostfit1, valid.matrix), validation$Profitability)

save(fit1, file = "model1.RData")
save(fit2, file = "model2.RData")
save(fit4, file = "model4.RData")
save(boostfit1, file = "boostmodel.RData")


###################### Submission Code###############################################


setwd("/home/fractaluser/Documents/DMA-Challenge")

library(data.table)
library(caret)
library(Matrix)
library(xgboost)
library(corrgram)
library(ggplot2)
library(ROCR)
library(pROC)
library(lubridate)
library(rpart)

test <- fread("EY_DMA_Analytics_2016_Testing_Data_0822.csv", stringsAsFactors = FALSE)

test$state_flag1 <- ifelse(test$addr_state %in% c("MS", "ME", "ND", "NE"), 1, 0) 

test$total_rev_hi_lim <- log(test$total_rev_hi_lim + 1)
test$total_rev_hi_lim[is.na(test$total_rev_hi_lim)] <- mean(test$total_rev_hi_lim, na.rm = TRUE)
test$total_rev_hi_lim_outlierflag <- ifelse(test$total_rev_hi_lim > 12, 1, 0)
test$total_rev_hi_lim[test$total_rev_hi_lim > 12] <- 12

test$tot_cur_bal <- log(test$tot_cur_bal+1)
test$tot_cur_bal[is.na(test$tot_cur_bal)] <- mean(test$tot_cur_bal, na.rm = TRUE)
test$tot_cur_bal[test$tot_cur_bal > 13.5] <- 13.5

test$annual_inc <- log(test$annual_inc + 1)
test$annual_inc[test$annual_inc > 13] <- 13


test$purpose <- factor(test$purpose)
test$emp_length <- factor(test$emp_length, levels = c("n/a","< 1 year","1 year",
                                                      "2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"))

test$term <- factor(test$term, levels = c(" 36 months", " 60 months"))
test$home_ownership[test$home_ownership %in% c("NONE", "OTHER", "ANY")] <- "NONE+OTHER+ANY"
test$home_ownership <- factor(test$home_ownership, levels = c("NONE+OTHER+ANY","MORTGAGE", "RENT", "OWN"))

test$verification_status <- factor(test$verification_status, levels = c("Not Confirmed","Confirmed", "Source Confirmed"))

test$mths_since_last_delinq[is.na(test$mths_since_last_delinq)] <- median(test$mths_since_last_delinq, na.rm = TRUE)
test$initial_list_status <- factor(test$initial_list_status)

test$issue_d <- paste0("01-", test$issue_d)
test$earliest_cr_line <- paste0("01-", test$earliest_cr_line)

test$issue_d <- dmy(test$issue_d)
test$earliest_cr_line <- dmy(test$earliest_cr_line)

test$partial_age <- as.numeric(test$issue_d - test$earliest_cr_line)/365
test$partial_age[is.na(test$partial_age)] <- median(test$partial_age, na.rm = TRUE)
test$partial_age[test$partial_age > 40] <- 40

test$collections_12_mths_ex_med[is.na(test$collections_12_mths_ex_med)] <- 0
test$collections_12_mths_ex_med[test$collections_12_mths_ex_med > 2] <- 2

test$dti[test$dti > 40] <- 40

test$open_acc[test$open_acc > 30] <- 30

test$pub_rec[test$pub_rec > 2] <- 2
test$pub_rec_flag <- ifelse(test$pub_rec >= 1, 1, 0)

test$revol_bal <- log(test$revol_bal + 1)
test$revol_bal[test$revol_bal > 11.5] <- 11.5

test$total_rec_late_fee_flag <- ifelse(test$total_rec_late_fee > 0, 1, 0)

test$desc_flag <- ifelse(test$desc=="", 1, 0)

test$issue_month <- month(test$issue_d)
test$issue_year <- year(test$issue_d)

test$last_credit_pull_d <- paste0("01-", test$last_credit_pull_d)
test$last_credit_pull_d <- dmy(test$last_credit_pull_d)

test$last_credit_pull_month <- month(test$last_credit_pull_d)
test$last_credit_pull_year <- year(test$last_credit_pull_d)

test$earliest_cr_line_month <- month(test$earliest_cr_line)
test$earliest_cr_line_year <- year(test$earliest_cr_line)

test$last_credit_pull_year[is.na(test$last_credit_pull_year)] <- 2016
test$last_credit_pull_month[is.na(test$last_credit_pull_month)] <- 1


test$last_pymnt_d <- paste0("01-", test$last_pymnt_d)
test$last_pymnt_d <- dmy(test$last_pymnt_d)

test$last_pymnt_month <- month(test$last_pymnt_d)
test$last_pymnt_year <- year(test$last_pymnt_d)

test$last_pymnt_year[is.na(test$last_pymnt_year)] <- 2016
test$last_pymnt_month[is.na(test$last_pymnt_month)] <- 1

# test$next_pymnt_d <- paste0("01-",test$next_pymnt_d)
# test$next_pymnt_d <- dmy(test$next_pymnt_d)
# test$next_pymnt_month <- month(test$next_pymnt_d)
# test$next_pymnt_month[is.na(test$next_pymnt_month)] <- 2

test$consolidation_flag <- 0
test$consolidation_flag[grep("consolid", test$title)] <- 1

########################################################################

test$issue_month_flag <- ifelse(test$issue_month==12, 1, 0)
test$last_pymnt_month_flag <- ifelse(test$last_pymnt_month %in% c(1,12), 1, 0)
test$last_credit_pull_month_flag <- ifelse(test$last_credit_pull_month==1, 1, 0)

#########################################################################################

test$loan_status_prob <- predict(fit1, test, type = "response")
test$loan_status_pred <- ifelse(test$loan_status_prob >=0.31, "Good", "Bad") 
test$loan_status_pred <- factor(test$loan_status_pred, levels = c("Bad", "Good"))

test$score <- (test$loan_status_prob * test$dti)*100

test$lm_pred <- predict(fit2, test)
test$lm_pred_flag <- 1
test$lm_pred_flag[test$Profitability < 0] <- -1

test$tree_pred <- predict(fit4, test)

test$purpose_flag1 <- ifelse(test$purpose %in% c("educational", "wedding"), 1, 0)
test$purpose_flag2 <- ifelse(test$purpose %in% c("small_business"), 1, 0)

test$verification_status_flag <- ifelse(test$verification_status=="Not Confirmed", 1, 0)
test$verification_status_joint_flag <- ifelse(test$verification_status_joint %in% c("Source Verified", "Verified"), 1, 0)


#################################################################################################

test <- data.frame(test)
test.small <- test[, c("loan_status_pred", "lm_pred", "tree_pred","loan_amnt", "funded_amnt", "funded_amnt_inv", "term",
                       "total_rec_late_fee", "state_flag1", "lm_pred_flag","purpose","purpose_flag1", "purpose_flag2",
                       "verification_status", "partial_age", "annual_inc", "issue_month", "issue_year",
                       "last_credit_pull_month", "last_credit_pull_year","mths_since_last_delinq",
                       "last_pymnt_month", "last_pymnt_year", "score", 
                       "issue_month_flag", "last_pymnt_month_flag","last_credit_pull_month_flag")] 

test.small$Profitability <- 1

test.matrix <- sparse.model.matrix(Profitability ~ ., data = test.small) 

test$Profitability <- predict(boostfit1, test.matrix)

submission <- data.frame(id = test$id, predicted_profitability = test$Profitability)

write.csv(submission, "submit.csv", row.names = FALSE)

################################################################################################################