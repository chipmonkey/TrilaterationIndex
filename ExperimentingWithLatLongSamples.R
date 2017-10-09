sampledata <- read.csv('data/lat_long_synthetic.csv')
head(sampledata)
library(nnclust)
# install_github('chipmonkey/nnclust')

mynn <- nncluster(data.matrix(sampledata), threshold = 0.2)


head(mynn)
summary(mynn)
sapply(mynn[[1]], class)
head(mynn[[1]]$mst)


x <- nnfind(data.matrix(sampledata))
head(x$dist)
# Is this right?

plot(x$neighbour[1:1000])

