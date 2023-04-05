library(ROCR)

rocplot <- function(pred, truth, ...) {
    predob <- prediction(pred, truth)
    perf <- performance(predob, "tpr", "fpr")
    plot(perf, ...)
}

set.seed(1)

x <- matrix(rnorm(200 * 2), ncol = 2)
x[1:100, ] <- x[1:100, ] + 2
x[101:150, ] <- x[101:150, ] - 2
y <- c(rep(1, 150), rep(2, 50))

dat <- data.frame(x = x, y = as.factor(y))

train <- sample(200, 100)

svmfit <- svm(y ~ ., data = dat[train, ], kernel = "radial",
              gamma = 2, cost = 1,
              decision.values = TRUE)
fitted <- attributes(
        predict(svmfit, dat[train, ], decision.values = TRUE)
        )$decision.values

par(mfrow = c(1, 2))
rocplot(-fitted, dat[train, "y"], main = "Training Data")

svmfit_ <- svm(y ~ ., data = dat[train, ], kernel = "radial",
              gamma = 50, cost = 1,
              decision.values = TRUE)
fitted_ <- attributes(
        predict(svmfit_, dat[train, ], decision.values = TRUE)
        )$decision.values
rocplot(-fitted_, dat[train, "y"], main = "Training Data")

fitted <- attributes(
            predict(svmfit, dat[-train, ], decision.values = TRUE)
            )$decision.values
rocplot(-fitted, dat[-train, "y"])

fitted_ <- attributes(
            predict(svmfit_, dat[-train, ], decision.values = TRUE)
            )$decision.values
rocplot(-fitted_, dat[-train, "y"])
