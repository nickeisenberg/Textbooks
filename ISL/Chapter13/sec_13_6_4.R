# Re-Sampling approch for the Khan dataset
library(ISLR2)
attach(Khan)
x <- rbind(xtrain, xtest)
y <- c(as.numeric(ytrain), as.numeric(ytest))

dim(x)

table(y)

# There are four classes of cancer. We will compare the mean expression
# in the second class to the mean expression in the fourth class. We will
# perform the t-test in the 11th gene.
x <- as.matrix(x)
x1 <- x[which(y == 2), ]
x2 <- x[which(y == 4), ]
n1 <- nrow(x1)
n2 <- nrow(x2)
t_out <- t.test(x1[, 11], x2[, 11], var.equal = TRUE)
TT <- t_out$statistic

TT
t_out

# However this p-value relies on the assumption that under the null hypothesis
# of no difference betweenthe two groups, the test statistic follows a
# t-distribution. We can know use sampleing to randomly assign the two groups
# into groups of size 29 and 25 under the assumption that the distribution in
# each group is the same. We can repeat this processes 10000 times. We record
# the number of times that our test statistic exceeds the test statistcs
# obtained via resampling.

set.seed(1)
B <- 10000
Tbs <- rep(NA, B)
for (b in 1:B) {
    dat <- sample(c(x1[, 11], x2[, 11]))
    Tbs[b] <- t.test(dat[1:n1], dat[(n1 + 1): (n1 + n2)],
                     var.equal = TRUE
                     )$statistic
}
mean((abs(Tbs) >= abs(TT)))

hist(Tbs, breaks = 100, xlim = c(-4.2, 4.2), main = "",
     xlab = "Null Distribution of Test Statistic", col = 7)
lines(seq(-4.2, 4.2, len = 1000),
      dt(seq(-4.2, 4.2, len = 1000),
         df = (n1 + n2 - 2)
         ) * 1000, col = 2, lwd = 3)
abline(v = TT, col = 4, lwd = 2)
text(TT + .5, 350, paste("T = ", round(TT, 4), sep = ""),
     col = 4)

# Now we implement the resampling FDR approch outlined in the algorithm 13.4.
# We will do this approch on a random subset of 100 genes of the 2308 genes.

B <- 500
m <- 100
set.seed(1)
index <- sample(ncol(x1), m)
Ts <- rep(NA, m)
Ts_star <- matrix(NA, ncol = m, nrow = B)
for (j in 1:m){
    k <- index[j]
    Ts[j] <- t.test(x1[, k], x2[, k],
                    var.equal = TRUE
                    )$statistic
    for (b in 1:B) {
        dat <- sample(c(x1[, k], x2[, k]))
        Ts_star[b, j] <- t.test(dat[1:n1],
                                dat[(n1 + 1): (n1 + n2)],
                                var.equal = TRUE
                                )$statistic
    }
}

cs <- sort(abs(Ts))
FDRs <- Rs <- Vs <- rep(NA, m)
for (j in 1:m) {
    R <- sum(abs(Ts) >= cs[j])
    V <- sum(abs(Ts_star) >= cs[j]) / B
    Rs[j] <- R
    Vs[j] <- V
    FDRs[j] <- V
    FDRs[j] <- V / R
}

max(Rs[FDRs <= .1])
sort(index[abs(Ts) >= min(cs[FDRs < .1])])
max(Rs[FDRs <= .2])
sort(index[abs(Ts) >= min(cs[FDRs < .2])])

plot(Rs, FDRs, xlab = "Number oif Rejections", type = "l",
     ylab = "False discovery rate", col = 4, lwd = 3)
