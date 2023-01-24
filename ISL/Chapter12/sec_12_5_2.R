X <- data.matrix(scale(USArrests))

pcob <- prcomp(X)
summary(pcob)

sX <- svd(X)
names(sX)

round(sX$v, 3)

pcob$rotation

t(sX$d * t(sX$u))

pcob$x

sX$d
