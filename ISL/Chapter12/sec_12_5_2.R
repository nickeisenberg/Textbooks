X <- data.matrix(scale(USArrests))

pcob <- prcomp(X)
summary(pcob)

sX <- svd(X)
names(sX)

pcob$rotation

# u has the standardized pca scores.
# d has the standard deviations of the unstadardized version of u.
# To unstandardize u, multiply by d
sX$d
sX$u
t(sX$d * t(sX$u))

# omit 20 random entries in the matrix X so we can use PCA to try and recover.
nomit <- 20
set.seed(15)
ina <- sample(seq(50), nomit)
inb <- sample(1:4, nomit, replace = TRUE)
Xna <- X
index_na <- cbind(ina, inb)
Xna[index_na] <- NA

# PCA function for estimation
fit_svd <- function(X, M = 1){
    svdob <- svd(X)
    with(svdob,
         u[, 1:M, drop = FALSE] %*%
             (d[1:M] * t(v[, 1:M, drop = FALSE]))
         )
}

# step one of the algorithim
Xhat <- Xna
xbar <- colMeans(Xhat, na.rm = TRUE)
Xhat[index_na] <- xbar[inb]

# set up some threshold values
thresh <- 1e-7
rel_err <- 1
iter <- 0
ismiss <- is.na(Xna)
mssold <- mean((scale(Xna, xbar, FALSE)[!ismiss])^2)
mss0 <- mean(Xna[!ismiss]^2)

# set up the algorithim
while(rel_err > thresh){
    iter <- iter + 1
    Xapp <- fit_svd(Xhat, M = 1)
    Xhat[ismiss] <- Xapp[ismiss]
    mss <- mean(((Xna - Xapp)[!ismiss])^2)
    rel_err <- (mssold - mss) / mss0
    mssold <- mss
    cat("Iter:", iter, "MSS:", mss, "Rel Err:", rel_err, "\n")
}

cor(Xapp[ismiss], X[ismiss])

