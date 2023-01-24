ltibrary(ISLR2)

nci_labs <- NCI60$labs
nci_data <- NCI60$data

dim(nci_data)
nci_labs

table(nci_labs)

pr_out <- prcomp(nci_data, scale = TRUE)

Cols <- function(vec) {
    cols <- rainbow(length(unique(vec)))
    return(cols[as.numeric(as.factor(vec))])
}
Cols(nci_labs)

par(mfrow = c(1, 2))
plot(pr_out$x[, 1:2], col = Cols(nci_labs), pch = 19,
     xlab = "Z1", ylab = "Z2")
plot(pr_out$x[, c(1, 3)], col = Cols(nci_labs), pch = 19,
     xlab = "Z1", ylab = "Z2")

summary(pr_out)

plot(pr_out)

pve <- 100 * pr_out$sdev^2 / sum(pr_out$sdev^2)
par(mfrow = c(1, 2))
plot(pve, type = "o", ylab = "PVE",
     xlab = "Principle Component", col = "blue")
plot(cumsum(pve), type = "o", ylab = "Cumulative PVE",
     xlab = "Principle Component", col = "brown")

summary(pr_out)$importance[2, ]

# Clustering the cancer types using hierarchical clustering

sd_data <- scale(nci_data)

par(mfrow = c(1, 3))
data_dist = dist(sd_data)
plot(hclust(data_dist), xlab = "", sub = "", ylab = "",
     labels = nci_labs, main = "Complete Linkage")
plot(hclust(data_dist, method = "average"),
     labels = nci_labs, main = "Average Linkage")
plot(hclust(data_dist, method = "single"),
     labels = nci_labs, main = "Single Linkage")

hc_out <- hclust(dist(sd_data))
hc_clusters <- cutree(hc_out, 4)
table(hc_clusters, nci_labs)

par(mfrow = c(1, 1))
plot(hc_out, labels = nci_labs)
abline(h = 139, col = "red")

hc_out

# compare kmeans with hierarchical clustering
set.seed(2)
km_out <- kmeans(sd_data, 4, nstart = 20)
km_clusters <- km_out$cluster
table(km_clusters, hc_clusters)

# hclust on the first few principle components
hc_out <- hclust(dist(pr_out$x[, 1:5]))
plot(hc_out, labels = nci_labs,
     main = "Hier. CLustering on the first five score vectors")
table(cutree(hc_out, 4), nci_labs)

# kmeans on first few principle components
km_out <- kmeans(dist(pr_out$x[, 1:5]), 4, 20)
table(km_out$cluster, nci_labs)
