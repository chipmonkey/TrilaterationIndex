library(ggplot2)

data <- read.csv('./data/small_sample.csv')
data <- data[,-1]
data

refpoints <- read.csv('./data/ref_points.csv')
refpoints <- refpoints[,-1]
names(refpoints) <- c('A', 'x', 'y')
refpoints

querypoint <- data.frame(array(c(50, 80), dim=c(1,2)))
names(querypoint) <- c('x', 'y')
querypoint

ggplot() +
  geom_point(data=data, mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3)


