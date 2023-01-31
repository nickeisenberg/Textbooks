# Now we will perfrom the hypothesis test for all 2000 fund managers in the
# Fund dataset.
library(ISLR2)
fund_pvalues <- rep(0, 2000)
for (i in 1:2000)
    fund_pvalues[i] <- t.test(Fund[, i], mu = 0)$p.value

# In this example there are too mant fund managers to try to control
# the FWER so instead we control the FDR by using the benjamin-hochberg
# procedure
q_values <- p.adjust(fund_pvalues, method = "BH")
q_values[1:10]

# For q level if %10
sum(q_values <= .1)

# Go through the benjamin-hochberg procedure
p_sorted <- sort(fund_pvalues)
m <- length(p_sorted)
q <- .1
which_p_sorted <- which(p_sorted < q * (1:m) / m)
if (length(which_p_sorted) > 0) {
    wh <- 1:max(which_p_sorted)
} else {
    wh <- numeric(0)
}

plot(p_sorted, log = "xy", ylim = c(4e-6, 1), ylab = "P-value")
points(wh, p_sorted[wh], col = 4)
abline(a = 0, b = q / m, col = 2, untf = TRUE)
abline(h = .1 / 2000, col = 3)
