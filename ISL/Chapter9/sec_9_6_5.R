library(ISLR2)

names(Khan)

dim(Khan$xtrain)
length(Khan$ytrain)

dim(Khan$xtest)
length(Khan$ytest)

table(Khan$ytrain)
table(Khan$ytest)

dat <- data.frame(
        x = Khan$xtrain,
        y = as.factor(Khan$ytrain)
        )

out <- svm(y ~ ., data = dat, kernel = "linear",
           cost = 10)
summary(out)

table(out$fitted, dat$y)

dat_test <- data.frame(
            x = Khan$xtest,
            y = as.factor(Khan$ytest))
pred_te <- predict(out, newdata = dat_test)
table(pred_te, dat_test$y)
