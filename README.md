# Total-profit-improving-using-ML
# Setup
* Python 3.7

# Abstract
The code provides the explanation of how to use machine learning to maximize the total profit generated by managing non-customers with a credit card offer in a bank company, specially using datasets from the datathon international Interbank in which I finished sixth. Link: [Interbank-internacional-2019](https://www.kaggle.com/c/interbank-internacional-2019/overview). Using stratified K-Fold and lightgbm framework provide the solution in this project. Finally using a differential evolution algorithm the cut off point is estimated maximizing the total profit. 

# Introduction
In this section, the target (output) is generated and analyzed. The <b>codtarget</b> is a binary variable that indicates if a customer contacted accept the credit card offer or not, but that doesn´t indicate the use of it. If a customer accept the credit card but he doesn´t use it, the bank generates a management cost with no return. The margen indicate the expected profit of a customer, so the new variable is generated from the margen variable taking Margen>0 = 1 (as an event of interest) and Margen<=0 = 0. Note that the target is the intersection of events: a person acquires a credit card and the person generates profitability.

<img src="resource/plots/codtarget_graph.png" height="80%" width="80%"/>
