# kd tree
library(RANN)

data <- read.csv('./data/point_sample_10.csv')
data <- data[,-1]
data

refpoints <- read.csv('./data/sample_ref_points.csv')
refpoints <- refpoints[,-1]
names(refpoints) <- c('A', 'x', 'y')
refpoints

querypoint <- data.frame(array(c(50, 65), dim=c(1,2)))
names(querypoint) <- c('x', 'y')
querypoint

mytree <- kdtree(data)
?mathart
??kdtree

system.time(nearest <- nn2(data, data))

myPoints <- makepoints(100000, 0, 10000)
system.time(nearest <- nn2(myPoints, data))

nearest
summary(nearest$nn.dists)
