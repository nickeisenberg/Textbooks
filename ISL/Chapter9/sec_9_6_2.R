library(e1071)

# Generate some data with a nonlinear class boundary
set.seed(1)

x <- matrix(rnorm(200 * 2), ncol = 2)
x[1:100, ] <- x[1:100, ] + 2
x[101:150, ] <- x[101:150, ] - 2
y <- c(rep(1, 150), rep(2, 50))

dat <- data.frame(x = x, y = as.factor(y))

plot(x, col = y)

# We now randomly split the data into training and test groups.
# We then fit that data using the svm() function with a radial kernal.
train <- sample(200, 100)
svmfit <- svm(y ~ ., data = dat[train, ], kernel = "radial",
              gamma = 1, cost = 1)
plot(svmfit, dat[train, ])

summary(svmfit)

# Increading the cost will increase the accuracy but lead to overfitting.
svmfit <- svm(y ~ ., data = dat[train, ], kernel = "radial",
              gamma = 1, cost = 1e5)
plot(svmfit, dat[train, ])

# We can use the tune function to select the best choice of gamma and cost
# for an SVM with radial kernel
tune_out <- tune(svm, y ~ ., data = dat[train, ],
                 kernel = "radial",
                 ranges = list(
                               cost = c(.1, 1, 10, 100, 1000),
                               gamma = c(.5, 1, 2, 3, 4)
                               )
                 )

summary(tune_out)

# We see that the best choice of parameters is cost = 1 and gamma = .5
table(true = dat[-train, "y"],
      pred = predict(tune_out$best.model,
                     newdata = dat[-train, ]
                     )
      )

