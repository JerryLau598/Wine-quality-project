library(ggplot2)
library(tree)
library(psych)
library(dplyr)
library(car)
library(MASS)
library(reshape2)
library(ISLR)
library(boot)
library(caTools)

#We first come to input our data set, named Red and White
Red <- read.csv('/Users/liuxuetao/RStudio/Project/winequality-red.csv', sep = ';',
                header = TRUE)
White <- read.csv('/Users/liuxuetao/RStudio/Project/winequality-white.csv', sep = ';',
                  header = TRUE)
#Then we come to check whether there are some missing values
sum(is.na(Red))
sum(is.na(White))

#PCA
fa.parallel(Red[, 1:11], fa = 'pc', n.iter = 1000, show.legend = TRUE)
red_pca = principal(Red[, 1:11], nfactors = 4, rotate = 'none')
red_pca$weights
red_pca$scores
summary(red_pca)

#We therefore can gain our new dataset
red_new = red_pca$scores
red_new = cbind(red_new, quality = Red$quality)
red_new = data.frame(red_new)

#We then come to see whether linear regression can fit the data
set.seed(12315)
index = sample(nrow(red_new), floor(nrow(red_new)*0.8), replace = F )
train_red = red_new[index, ]
test_red = red_new[-index, ]

plot(train_red$PC1, train_red$quality)

red.lm = lm(quality ~. - quality2, data = train_red)
summary(red.lm)
par(mfrow=c(2,2), oma = c(1,1,0,0) + 0.1, mar = c(3,3,1,1) + 0.1)
plot(red.lm)

#Calculate the accuracy rate
predlm_red = predict(red.lm, train_red)
predlm_red = round(predlm_red)
head(predlm_red)
#Create the confusion matrix
lm_tab = table(predicted = predlm_red, actual = train_red$quality)
lm_tab
lm_tab = lm_tab[, c(-1)]
lm_tab = lm_tab[, -ncol(lm_tab)]
lm_tab
#Come to calculate the accuracy rate
sum(diag(lm_tab))/length(train_red$quality)


#Test and validation
predlm_red_t = predict(red.lm, test_red)
predlm_red_t = round(predlm_red_t)
head(predlm_red)
#Create the confusion matrix
lm_tab_t = table(predicted = predlm_red_t, actual = test_red$quality)
lm_tab_t
lm_tab_t = lm_tab_t[, c(-1)]
lm_tab_t = lm_tab_t[, c(-ncol(lm_tab_t))]
lm_tab_t
#Come to calculate the accuracy rate
sum(diag(lm_tab_t))/length(test_red$quality)

#We then come to use polynomial regression


# container of test errors
cv.RedMSE_poly = NA

# loop over powers of all variables
for (i in 1:15) {
  red.poly =  glm(quality ~ poly(PC1, i)
                  + poly(PC2, i) + poly(PC3, i)
                  + poly(PC4, i), data = red_new)
  # we use cv.glm‘s cross-validation and keep the vanilla cv test error
  cv.RedMSE_poly[i] = cv.glm(red_new, red.poly, K = 10)$delta[1]
}
# inspect results object
cv.RedMSE_poly

# illustrate results with a line plot connecting the cv.error dots
plot( x = 1:15, y = cv.RedMSE_poly, xlab = "power of multivariate",
      ylab = "CV error", type = "b", pch = 19, lwd = 2, bty = "n", 
      ylim = c( min(cv.RedMSE_poly) - sd(cv.RedMSE_poly), max(cv.RedMSE_poly) + sd(cv.RedMSE_poly) ) )
# horizontal line for 1se to less complexity
abline(h = min(cv.RedMSE_poly) + sd(cv.RedMSE_poly) , lty = "dotted")

# where is the minimum
points( x = which.min(cv.RedMSE_poly), y = min(cv.RedMSE_poly), col = "red", pch = "X", cex = 1.5 )

#It means the quadratic polynomial has the lowest MSE

#We then use degree 2 to fit the data
poly.fit = lm(quality ~ poly(PC1, 3) + poly(PC2, 3)
          + poly(PC3, 3) + poly(PC4, 3), data = red_new)
poly.preds = predict(poly.fit, red_new)
poly.preds = round(poly.preds)
head(poly.preds)

poly_tab = table(predicted = poly.preds, actual = red_new$quality)
poly_tab
poly_tab = poly_tab[, c(-1)]
poly_tab = poly_tab[, c(-ncol(poly_tab))]
poly_tab
#Come to calculate the accuracy rate
sum(diag(poly_tab))/length(red_new$quality)


#Then we come to use logistic regression
#When conducting the linear regression

red_new$quality2 = as.factor(red_new$quality)
#Since we want to use logistic regression, we need the catagorical variable

set.seed(10086)
index = sample(nrow(red_new), floor(nrow(red_new)*0.8), replace = F )
train_red_1 = red_new[index, ]
test_red_1 = red_new[-index, ]

head(train_red_1)

#We now come to fit our model
# Hess=TRUE to let the model output show the observed information matrix from optimization which is used to get standard errors.
o_lrm = polr(quality2 ~ . - quality, data = train_red_1, Hess=TRUE)
vif(o_lrm)
summary(o_lrm)

olrm.pred = predict(o_lrm, type = "class")
head(olrm.pred)

#Confusion matrix
cm1 = as.matrix(table(Actual = train_red_1$quality2, Predicted = olrm.pred))
cm1

sum(diag(cm1))/length(train_red_1$quality2)

#Now we use test data
olrt.pred = predict(o_lrm, newdata = test_red_1, type = "class")
cm2 = as.matrix(table(Actual = test_red_1$quality2, Predicted = olrt.pred))
cm2
sum(diag(cm2))/length(test_red_1$quality2)

#Now we come to see whether stepwise method can increase our accuracy rate

o_lr = step(o_lrm)

#The result shows that we have not wiped out any variables

ggplot(data = Red, aes(x = quality, fill = quality)) +
  geom_bar(width = 1, color = 'black',fill = I('#F596AA'))


#Binomial Logistic Regression
red_new$category[red_new$quality <= 5] = 0
red_new$category[red_new$quality > 5] = 1

red_new$category = as.factor(red_new$category)
head(red_new)

ggplot(data = red_new, aes(x = category)) +
  geom_bar(width = 1, color = 'black',fill = I('#F596AA'))

set.seed(10086)
index = sample(nrow(red_new), floor(nrow(red_new)*0.8), replace = F )
train_red_2 = red_new[index, ]
test_red_2 = red_new[-index, ]

red_glm = glm(category ~ . - quality - quality2, 
               data = train_red_2, family=binomial(link = "logit"))
red_glm

head(fitted(red_glm))

red_binom_pred = ifelse(predict(red_glm, type = "response") > 0.5,"Good Wine", "Bad Wine")
head(red_binom_pred)

binom_tab = table(predicted = red_binom_pred, actual = train_red_2$category)
binom_tab

sum(diag(binom_tab))/length(train_red_2$category)

tst_pred = ifelse(predict(red_glm, newdata = test_red_2,
                          type = "response") > 0.5, "Good Wine", "Bad Wine")
tst_tab = table(predicted = tst_pred, actual = test_red_2$category)
tst_tab

sum(diag(tst_tab))/length(test_red_2$category)
#We then use test set

############################################
############################################
############################################

#Unsupervised learning

#Since we want to use the unsupervised learning, we do not need to
#assign each categories with some label, i.e, quality, but to let
#our machine do it automatically.
library(ISLR)
library(tree)
library(e1071)

set.seed(12306)
index = sample(nrow(Red), floor(nrow(Red)*0.8), replace = F )
train.red = Red[index, ]
test.red = Red[-index, ]

#We first come to fit a classification tree
tree.red = tree(as.factor(quality) ~., data = Red, subset = index)
summary(tree.red)
plot(tree.red, main = "Classification Tree")
text(tree.red, pretty = 0)

tree.red

tree.pred = predict(tree.red, train.red, type = "class")
cm3 = table(tree.pred, train.red$quality)
cm3
sum(diag(cm3))/length(train.red$quality)

tree.pred = predict(tree.red, test.red, type = "class")
cm3 = table(tree.pred, test.red$quality)
cm3
sum(diag(cm3))/length(test.red$quality)
#因为我们的dataset大部分都在中间，所以准确率偏高
#Now we come to see whether we can improve our model or not
cv.red = cv.tree(tree.red, FUN = prune.misclass)
cv.red

plot(cv.red$size, cv.red$dev, type = "b")

#Let's try prune at 4 and 8 
prune.red1 = prune.misclass(tree.red, best = 4)
plot(prune.red1)
text(prune.red1, pretty = 0)
summary(prune.red1)$misclass

tree.pred1 = predict(prune.red1, test.red, type = "class")
cm4 = table(tree.pred1, test.red$quality)
cm4
sum(diag(cm4))/length(test.red$quality)

#prune at 8

prune.red2 = prune.misclass(tree.red, best = 8)
plot(prune.red2)
text(prune.red2, pretty = 0)
summary(prune.red2)$misclass

tree.pred2 = predict(prune.red2, test.red, type = "class")
cm5 = table(tree.pred2, test.red$quality)
cm5
sum(diag(cm5))/length(test.red$quality)
#准确率没变


#Let us try the regression tree
tree.red_reg = tree(quality ~., data = Red, subset = index)
summary(tree.red_reg)
plot(tree.red_reg)
text(tree.red_reg, pretty = 0)

tree.red_reg

tree_reg.pred = predict(tree.red_reg, test.red)
tree_reg.pred = round(tree_reg.pred)
cm6 = table(tree_reg.pred, test.red$quality)
cm6 = cm6[, -1]
cm6 = cm6[, -ncol(cm6)]
sum(diag(cm6))/length(test.red$quality)
#因为我们的dataset大部分都在中间，所以准确率偏高
#Now we come to see whether we can improve our model or not
cv.red_reg = cv.tree(tree.red_reg)
cv.red_reg

plot(cv.red_reg$size, cv.red_reg$dev, type = "b")


#We then come to try the bagging method
library(randomForest)

set.seed(10697)
bag.red = randomForest(quality~., data = Red, 
                      subset = index, mtry=11,ntree=5000,
                      importance=T) 
#We consider all the 11 variable
bag.red
importance(bag.red)
varImpPlot(bag.red)

bag.pred = predict(bag.red, newdata = Red[-index,])
bag.pred = round(bag.pred)
cm7 = table(bag.pred, test.red$quality)
cm7
cm7 = cm7[, -1]
cm7 = cm7[, -ncol(cm7)]
sum(diag(cm7))/length(test.red$quality)

#Then we come to try the random forest method
set.seed(10697)
rf.red = randomForest(quality~., data = Red, subset = index, 
                     mtry=6, ntree = 5000, importance=T)
rf.red
rf.pred = predict(rf.red, newdata = Red[-index,])
rf.pred = round(rf.pred)
cm8 = table(rf.pred, test.red$quality)
cm8

cm8 = cm8[, c(-1)]
cm8 = cm8[, -ncol(cm8)]
cm8
sum(diag(cm8))/length(test.red$quality)
