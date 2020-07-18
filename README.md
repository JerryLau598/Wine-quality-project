# Wine-quality-project
To see whether human quality of tasting can be related to the chemical properties of wine. This is a final project of module called Applied Statistics, since it has 10 pages limit, we would only focus on the red wine data set, we can do white in the same way.

# Project description

Wine industry shows a recent growth spurt as social drinking is on the rise. The price of wine depends on a rather abstract concept of wine appreciation by wine tasters. Pricing of wine depends on such a volatile factor to some extent. Another key factor in wine certification and quality assessment is physicochemical tests which are laboratory- based and takes into account factors like acidity, pH level, presence of sugar and other chemical properties. For the wine market, it would be of interest if human quality of tasting can be related to the chemical properties of wine so that certification and quality assessment and assurance process is more controlled. Two data sets are available from https://archive.ics.uci.edu/ml/datasets/Wine+Quality of which one data set is on red wine and have 1599 different varieties and the other is on white wine and has 4898 varieties. All wines are produced in a particular area of Portugal. Data are collected on 11 different properties of the wines based on chemical including density, acidity, alcohol content etc. All chemical properties of wines are continuous variables. The last column is quality, which is an ordinal variable with possible ranking from 1 (worst) to 10 (best). Each variety of wine is tasted by three independent tasters and the final rank assigned is the median rank given by the tasters. Details of the variables involving chemical properties can be obtained from the data website.

# Introduction
With the booming of wine industry, we are not only focusing ”can we have some wine?”, but ”can we have some good wine?”. In this project, we would using both supervised and unsupervised learning to predict the quality of human tasting by some chemical factors like acidity, pH level and so on. When conducting supervised learning method such as multiple linear regression, polynomial regression and multiple logistic regression, we first implement the **Principal Component Analysis(PCA)** to our data set, which can reduce the dimension of our dataset and eliminate the collinearity. By doing so, we obtain a very good data set, where each principal component are independent each other and has only four of them, which significantly reduces our calculation. In unsupervised learning, we first try the Classification Tree and Regression Tree to predict the quality of the wine, then we implement the Bagging Method and Random Forest Method as well and calculate all the accuracy rate resprctively.


# Pre-processing
At the very first beginning, what we need to do is to import our data set. When we open the data set, we found out that it is not a ”comma” separated value, but ”semi-colon” separated value. Hence when we import the csv file by ```read.csv()``` function, we need to set the parameter ```sep = ’;’``` to import our data set correctly. We named our red wine and white wine data set by Red and White respectively. Then, we need to check whether we have any missing values, an convenient way to doing so is to sum up all the missing value and found that there are no missing value any more in our data set.

In this project, we would mainly focus on the **red wine** data set, since we have 10 pages limit, and it would be so rush if we do both, so we choose to focus on red wine data set. In supervised learning, as we said before, we want to conduct the PCA first, but there is a question, how many principal component we should use? In order to solve this problem, we draw the scree plots to choose the optimal number of principal components as Fig 1. ![fig1](https://i.ibb.co/xjVCRng/fg1.png)

Since the intersection of our resampled data and actual data is between four and five principal component numbers, we therefore choose four be the number of our principal component numbers, since we need the eigenvalue of our principal components of our actual data greater than the resampled data’s. Then, we use `principal()` function to implement our PCA using four principal components, the result shows that we need four principal components to nearly explained all the variables, which is quite significant. After that, we need to construct our training set and test set to increase the accuracy of our model. We let 80% of the data be training set, rest of them be the test set. These training and test set are only valid for our supervised learning methods, since we would use original data set in unsupervised learning.

# Data Analysus
## Supervised Learning
### Multiple Linear Regression

In this section, we try to use multiple regression to our new data set. Since we eliminate all the collinearity in our data set, we can just simply use `lm()` function to implement the multiple linear regression. The results shows that adjusted R-squared is only **0.3222**. Since our adjusted R-squared is very small, which means our model is not very good, so we want to see how good we can make by calculating the accuracy rate. We first use `predict()` function to predict the quality of our wine by our four principal components, then we use `round()` function to round up or round down the results, since quality of the wine are integers, but the results obtained from predict() is not. The accuracy rate we calculated is **0.5551212**. *It is a little bit of strange that the model with adjusted R-squared 0.3222 would have nearly 55 percent accuracy rate, we will find out why this happen later*. We then try to use the test set to calculate the confusion matrix and accuracy, the result tells that the accuracy rate is **0.575**, even higher than our training set, which is also very strange and we would discuss it later.

### Multiple Polynomial Regression

The adjusted R-squared in our multiple linear regression is only 0.3222, so we wonder whether polynomial regression can improve our model. But we do not know which degree we should use, so we then use **cross - validation** to select our optimal polynomial degree d by choosing the smallest MSE amongst all the polynomial degree. The result of our CV process is given in the Fig 2 and the optimal degree is 3. ![fig2](https://i.ibb.co/9GZ0x9n/fg2.png)

Then, we perform a multiple polynomial regression with degree 3 and obtain the accuracy rate is **0.5653533**, slightly greater than the accuracy rate in multiple linear regression using training set.

### Multiple Logistic Regression

In order to implement the logistic regression, we need the variables in our data set are independent with each other, which is obvious in our data set, since each principal components are independent with each other. Then, we need to add a new column of variable, `quality2`, where we transfer the data type of quality from numeric to factor. Therefore, we also need to create a new training set and test set as well. Using `polr()` function, we conduct a multiple logistic regression on our new data set and the accuracy rate of it is **0.5715403**. Then, we use our test set to calculate the accuracy rate, which is given by **0.565625**.

Besides that, we want to see whether stepwise method can improve our model, using `step()` function and the result tells us that we do not need to eliminate any variables. When we checking our confusion matrices obtained by different methods, we found that nearly all the prediction are concentrated in **4, 5, 6**, so we want to see whether our data evenly distributed or not. Therefore, we plot the quality with respect to the number of each quality in Fig 3. ![fig3](https://i.ibb.co/DpMYrWj/fg3.png)

Unsurprisingly, nearly all the data points are concentrated in the quality **5, 6, 7**, which means **our data set has some flaws that its datas are not evenly distributed**. This would cause the accuracy rate higher than normal, since we can simply guess some number between 5, 6 or 7 and there is a good chance that we are correct. Therefore, we want to conduct a binomial logistic regression, where we separated our quality into two parts, bad when quality is less or equal to 5, and good otherwise. By doing so, we obtain an evenly distributed data set and it may increase our accuracy rate.

### Binomial Logistic Regression

We create a new variable called category, equals to 0 when its corresponding quality less or equal to 5, and 1 otherwise. Fig 4 shows that we obtain an evenly distributed data set and thus we can implement our binomial logistic regression. ![fig4](https://i.ibb.co/rQPGwPC/fg4.png)
The accuracy rate when using training set is **0.7232213** and **0.728125** when using test set. We can conclude that the accuracy rates are significantly improved when using binomial logistic regression.

## Unsupervised Learning
### Classification Tree

We then try to use unsupervised learning to predict the quality of wine using the given chemical properties. In this section, we would try both classification tree, where quality is factor here and regression tree, where we would predict first and then round them to integer. Using `tree()` function, we obtain out tree in Fig 5. ![fig5](https://i.ibb.co/tbLBsP8/fg5.png)

Noticed that we have 10 nodes in total and nearly all of them are concentrated in 5, 6 or 7, so we may use cross-validation to prune our tree to decrease our calculation. The result is given in the Fig 6. ![fig6](https://i.ibb.co/zbdsLcG/fg6.png)

Result shows that we can prune at 4 or 8 nodes, so we do the both and the accuracy is **0.578125** and **0.59375** respectively, hence we should prune at 8, since we would lost so many information when prune at 4.

### Regression Tree

When implementing regression tree method, we do not need to transfer our data type of quality into integer, but numeric, then we round our prediction into integer and calculate the accuracy rate. The regression tree is given in Fig 7. 
![fig7](https://i.ibb.co/nrs0hK1/fg9.png)

The corresponding accuracy rate is **0.5875**. As we conduct before, we apply cv to see whether we can prune our regression tree or not, and the resul tells us that we do not need to prune. The accuracy rate of our regression is **0.5875**, slightly lesser than classification tree.

### Bagging Method

The Bagging Method is a special situation of the Random Forest Method where we we consider all the variables in each iteration. In our simulation, we use **5000 iteration with 11 variables each**. After doing so, we obtain the results in Fig 8 
![fig8](https://i.ibb.co/6FwP22f/fg11.png)

This told us that alcohol is the most important variable among all the 11 variables and the accuracy calculating through test set is **0.68125**. What is more, the result shows that 46.26% of the variance can be explained by our model derived by Bagging Method.

### Random Forest Method

When conducting Random Forest Method to our data set, we remain 5000 iterations but only keep 6 variable in each iteration and the result tells us that our model can explained 46.76% of the total variance, and has 0.5% greater than the Bagging Method. The accuracy rate calculated using test set is **0.696875**, which is also slightly greater than Bagging Method.


# Conclusions

In this project, we use both supervised and unsupervised learning method to construct our model, and we calculate the accuracy rate in each method.

In supervised learning, what we do at the beginning is to conduct a **Principal Component Analysis** to decrease the dimension of our variables from 11 to 4, and also eliminate the collinearity. Then, we apply **Multiple Linear Regression, Multiple Polynomial Regression, Multiple Logistic Regression** and **Binomial Logistic Regression**. When implementing Multiple Polynomial Regression, we use Cross-Validation to select the optimal degree for our model, and the result shows that we should conduct a Multiple Polynomial Regression with degree 3. After that, we conduct a Multiple Logistic Regression, but the result shows that nearly all the prediction are concentrated in the quality 5, 6 or 7. Then we plot the quality with its total number and found out that **our data set are not evenly distributed**. Therefore, we create a new variable called category, to create an evenly distributed data set and implement a Binomial Logistic Regression. The accuracy rates using each method are given in the following table. 
||Multiple Linear Regression|Multiple Polynomial Regression|Multiple Logistic Regression|Binomial Logistic Regression|
|-|-|-|-|-|
|Accuracy Rate|0.575|0.5653533|0.565625|0.728125|

In unsupervised learning, we compare the accuracy rate of **Classification Tree** and **Regression Tree** and implement the **Bagging Method** and **Random Forest Method**. After conducting the Classifica- tion Tree, we found out that we can prune our tree from 10 nodes to 8 nodes and the accuracy rate is 0.59375. In Regression Tree, we cannot prune our tree and the accuracy rate is 0.5875, slightly lesser than the Classification Tree. When implementing the Bagging Method, we choose to use 5000 iteration and the result shows that our model can explained 46.26% of the total variance and has accuracy rate 0.68125, where Random Forest Method can explained 46.76% of the total variance and has accuracy rate 0.696875, both are slightly greater than Bagging Method. The accuracy rate using each unsupervised method are listed in the following table.
||Classification Tree|Regression Tree|Bagging Method|Random Forest Method|
|-|-|-|-|-|
|Accuracy Rate|0.59375|0.5875|0.68125|0.696875|

In conclusion, **the best model among all the methods is Binomial Logistic Regression** based on our new data set after Principal Component Analysis with accuracy rate **0.78125**. Though we have relatively high prediction accuracy rate, but using PCA data set would cause **lack of interpretability**. One of the difficulty we faced is that our data set are not evenly distributed, which would caused severely incorrect accuracy rate, and we would construct a better data set in the future if possible. The result of Bagging Method tells us that alcohol is the most important factor that makes a wine ”Good” or ”Bad”, then is sulphates and volatile acidity. There still has another strange place, in Fig 2, we found out that the CV error jumped at degree 13 and them come down and up again at further. Usually the CV error would decrease smoothly with the increase of the degree of polynomial, bu we haven’t figure out why this would happen.

