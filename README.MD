# KNN_Taiwan
Hi! With a purpose to practice my machine learning skills, I downloaded a dataset from here:
https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
The dataset was used for an article "The comparisons of data mining techniques for the predictive  
accuracy of probability of default of credit card clients" by I-Cheng Yeh and Che-hui Lien.
In the study they compared the performance of 6 different classification methods:
K-nearest neighbor, Logistic regression, Discriminant analysis, Naive Bayenesian, Neural networks and Classification trees. They concluded that "neural network is the only one that can accurately estimate the real probability of default."
I took the same dataset and tested out whether I would get the same performance results using the KNN method.

## Dataset information and preprocessing:
If you take a look at the dataset, you'll see that it is a mix of binary and continuous variables. I read that for KNN it is recommended to have variables in the same scale. So I used StandardScaler for the columns with numerical values so all columns would at least be in the same order of magnitude.

**X1**: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.  - **Applied StandardScaler**
**X2**: Gender (1 = male; 2 = female).  - **Replaced 2 with 0**
**X3**: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). - **Left it the same because the level of education seems gradual (the lower the score, the better the education**
**X4**: Marital status (1 = married; 2 = single; 3 = others).  - **Used binary dummy variables because no marital status is "better" than the others**
**X5**: Age (year).  - **Applied StandardScaler**
**X6 - X11**: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.  - **Applied StandardScaler**
**X12-X17**: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.  - **Applied StandardScaler**
**X18-X23**: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.- **Applied StandardScaler**

## Finding optimal K

In order to find an optimal K, all odd values from 1 to 50 were tried out when fitting the model. The smallest error rate was found at K = 37, but if you take a look at the graph, you could probably get a similar result by using 17, which is less computationally intensive.

![Finding optimal K value](https://github.com/henryginter/KNN_Taiwan/blob/master/K.png)

## Conclusion

With the value of K I achieved an error rate 0.183, which is similar to the achieved result found in the linked article (0.18).
I might continue this repo by finding other performance indicators such as area ratio and regression coefficient. Also at some point I'd like to give a shot of my own with ANN.


