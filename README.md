# DMA-Challenge (2016)

## Introduction

This repository is for publishing my solutions to DMA Analytics Challenge 2016 (officially sponsored by E&Y). I would like to thank Fractal Analytics for giving me an opportunity to participate in this competition. It was a great learning experience.

## Problem Description 

### Business Overview:
Company A is one of the world’s largest players in the online peer-to-peer lending business which has been instrumental in transforming the consumer and small business credit marketplace. The business model is as follows: the borrowers get access to lower interest rate loans through a fast and easy online or mobile interface, and investors provide the capital to enable many of the loans in exchange for earning interest. Since Company A uses little or no branch infrastructure, they can transfer the cost savings to the borrowers in form of lower interest rates and get attractive returns for the investors.
 
Company A is planning to use data analytics by leveraging the information on the existing loans that were extended to various consumers and small business and identifying the characteristics associated with the most highly profitable customers. For this purpose of this exercise, Company A has provided data on 768K loans that were issued in the past and the associated information captured, including:
 
·         Customer attributes at the time of application
·         Information around loan performance (last payment, outstanding balance, interest rate, etc.)
·         Loan status (Current, Delinquent, Charged off, etc.)
·         Some bureau attributes that were captured from the bureau data (past trade line information, etc.)
 
For Company A to increase its return on investments on a marketing campaign, it is important to understand the attributes that can help identify the most profitable customers in order to improve its solicitation as well as underwriting processes for new loans. From an analytical standpoint, there are 2 key aspects to an effective marketing campaign; a solicitation response model that helps to increase the acquisition rate and a high value customer identification model that helps the company get higher lifetime value from their customers. The focus of this problem is towards identifying the attributes of high value customers which can then be supplemented with a response rate model to enhance the ROI for a marketing investment.

### Analytics Challenge:
 
The dataset for training the model would consist of approx. 768K  loans which the participants would use to build their model to predict the estimated profitability of the loans. This consists of loans that existed on Company A’s books as of a point in time and has loans that are currently performing (i.e. in repayment), historically paid off and historically charged off loans. The target variable for the participants is the $ profitability associated with a loan, which is defined as the gross $ value margin that was made on that loan.
 
Subsequently, the validation dataset would have approx. 85K loans for which the participants will have to generate the prediction of the profitability for. The final evaluation for the competition will be defined by the Root Mean Squared Error (RMSE) of the predicted profitability values.

### Solution : 

My approach was ensembling of logistic model, linear model and single decision tree model. The predictions of all these models were fetched as an input to xgboost model which is used as final solution. 

## Results

I could not make to the top list though learned a lot through this and hoping to do better in next challenge. 

