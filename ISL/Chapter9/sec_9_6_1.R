library(e1071)

set.seed(1)
x <- matrix(rnorm(2 * 20), ncol = 2)
y <- c(rep(-1, 10), rep(1, 10))
x[y == 1,] <- x[y == 1, ] + 1

plot(x, col = (3 - y))

dat <- data.frame(x = x, y = as.factor(y))
svmfit <- svm(y ~ ., data = dat, kernel = "linear",
              cost = 10, scale = FALSE)
plot(svmfit, dat)

# indices for the support vectors
svmfit$index

summary(svmfit)

# same svm but with a smaller cost parameter
svmfit <- svm(y ~ ., data = dat, kernel = "linear",
              cost = .1, scale = FALSE)
plot(svmfit, dat)

svmfit$index

# use tune to perform cross validation on differenct cost parameters
tune_out <- tune(svm, y ~ ., data = dat, kernel = "linear",
                 ranges = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))

summary(tune_out)
names(tune_out)

bestmod <- tune_out$best.model
summary(bestmod)

# use the model to predict
x_test <- matrix(rnorm(2 * 20), ncol = 2)
y_test <- sample(c(-1, 1), 20, rep = TRUE)
x_test[y == 1,] <- x_test[y == 1,] + 1
testdat <- data.frame(x = x_test, y = as.factor(y))

y_pred <- predict(bestmod, testdat)
table(predict = y_pred, truth = testdat$y)

# use a cost of .01
svmfit <- svm(y ~ ., data = dat, kernel = "linear",
              cost = .01, truth = testdat$y)
ypred <- predict(svmfit, testdat)
table(predict = ypred, truth = testdat$y)

# Now lets further separate the clusters
x[y == 1,] <- x[y == 1,] +.5
plot(x, col = (y + 5) / 2, pch = 19)

dat <- data.frame(x = x, y = as.factor(y))
svmfit = svm(y ~ ., data = dat, kernel = "linear",
             cost = 1e5)
summary(svmfit)

plot(svmfit, dat)

# trying with a smaller cost
svmfit = svm(y ~ ., data = dat, kernel = "linear",
             cost = 1)
plot(svmfit, dat)
