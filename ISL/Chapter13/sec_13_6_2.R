# Family wise error rates for m = 1,..., 500 and alpha = .05, .01, and .001

m <- 1:500
fwe1 <- 1 - (1 - .05)^m
fwe2 <- 1 - (1 - .01)^m
fwe3 <- 1 - (1 - .001)^m

par(mfrow = c(1, 1))
plot(m, fwe1, type = "l", log = "x", ylim = c(0, 1), col = 2,
     ylab = "Family wise error rate", xlab = "Number of Hypotheses")
lines(m, fwe2, col = 4)
lines(m, fwe3, col = 4)
abline(h = .05, lty = 2)

# t-test for the first five manegers of the fund dataset
library(ISLR2)
fund_mini <- Fund[, 1:5]
fund_mini[1:5,]

# test the null hypothesis that the first fund managers mean returns was 0
t.test(fund_mini[, 1], mu = 0)

# Conduct the same test on the first 5 fund manegers
fund_pvalue <- rep(0, 5)
for (i in 1:5)
    fund_pvalue[i] <- t.test(fund_mini[, i], mu = 0)$p.value
fund_pvalue

# We need to adjust these values since we are testing multiple hypothesis
# bonferoni adjustment: this adjustment replaces each p-value
# with min(m * p, 1)
p.adjust(fund_pvalue, method = "bonferroni")
pmin(fund_pvalue * 5, 1)

# Holm adjustment: 
p.adjust(fund_pvalue, method = "holm")

# means of the managers returns
apply(fund_mini, 2, mean)

# We see that manager1 performs well where as manager2 performs poorly.
# Can we show that there is a statistical difference in performance?
t.test(fund_mini[, 1], fund_mini[, 2], paired = T)

# However recall that we performed this test only after observing the 
# first 5 tests. SO we need to account for the other 5C2 = 10 tests.
# For this we use TukeyHSD which takes as an inout a anova model
returns <- as.vector(as.matrix(fund_mini))
manager <- rep(c("1", "2", "3", "4", "5"), rep(50, 5))
a1 <- aov(returns ~ manager)
TukeyHSD(x = a1)

# We can plot the confidence interval of the pairwise comparisons
plot(TukeyHSD(x = a1))
