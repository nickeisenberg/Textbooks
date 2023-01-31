# Sim data for a t-test
# One group will be sampled from N(0,1) and the other N(.5, 1)
set.seed(6)
x <- matrix(rnorm(10 * 100), 10, 100)
x[, 1:50] <- x[, 1:50] + .5

# Test H_0: mu_1 = 0 (one-sample t-test)
t.test(x[, 1], mu = 0)

# The above t-test shows that we fail to refect even though we know mu_1 = 1/2
# Next we test H_0j: u_j = 0, j in {1, ..., 100}
p_values <- rep(0, 100)
for (i in 1:100)
    p_values[i] <- t.test(x[, i], mu = 0)$p.value
decision <- rep("Do not regect H0", 100)
decision[p_values <= .05] <- "Regect H0"
table(decision,
      c(rep("H0 is False", 50), rep("H0 is True", 50))
      )

# sim data with bigger difference in distribution
x <- matrix(rnorm(10 * 100), 10, 100)
x[, 1: 50] <- x[, 1:50] + 1
p_values = rep(0, 100)
for (i in 1:100)
    p_values[i] <- t.test(x[, i], mu = 0)$p.value
decision <- rep("Do not regect H0", 100)
decision[p_values <= .05] <- "Regect H0"
table(decision,
      c(rep("H0 is False", 50), rep("H0 is True", 50))
      )
