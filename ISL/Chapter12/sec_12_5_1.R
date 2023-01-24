states <- rownames(USArrests)

apply(USArrests, 2, mean)
apply(USArrests, 2, var)

pr_out <- prcomp(USArrests, scale = TRUE)

names(pr_out)

pr_out['center']
pr_out['scale']

dim(pr_out$rotation)
dim(pr_out$x)

biplot(pr_out, scale = 0)

pr_out$rotation <- -pr_out$rotation

biplot(pr_out, scale = 0)

pr_out$var <- pr_out$sdev^2

pve <- pr_out$var / sum(pr_out$var)
pve

par(mfrow = c(2, 1))
plot(pve, xlab = "Principle Component",
     ylab = "Proportion of Varience Explained",
     ylim = c(0, 1), type = "b")
plot(cumsum(pve), xlab = "Principle Component",
     ylab = "Cumulative Proportion of Varience Explained",
     ylim = c(0, 1), type = "b")
