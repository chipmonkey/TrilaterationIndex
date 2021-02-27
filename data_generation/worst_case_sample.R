# File paths are relative to this file's location, so if needed set pwd:
# setwd(getSrcDirectory()[1])
library(ggplot2)
# Distance function which I think we don't use:
myDist <- function(x, y) {
  # x and y must be 2D points -- (x,y) tuples i.e.: x <- c(1,2) y <- c(3,4)
  d <- sqrt((x[[1]]-y[[1]])^2+(x[[2]]-y[[2]])^2)
  return(d)
}

# Function to generate lots of random points
point_from_angle <- function(x, y, dist, theta) {
  myX = dist * cos(theta) + x
  myY = dist * sin(theta) + y
  result = c(myX, myY)
  return(result)
}

fade <- function(x, y, dist0, dist1, theta0, theta1, num_points) {
  dtheta <- (theta1 - theta0) / num_points
  ddist <- (dist1 - dist0) / num_points
  results <- data.frame(x=double(), y=double())
  for(i in 1:num_points) {
    newpoint <- array(point_from_angle(x, y, dist0 + ddist * i, theta0 + dtheta * i), dim=c(1,2))
    results[nrow(results)+1,] <- newpoint
  }
  return(results)
}

querypoint <- data.frame(array(c(50, 65), dim=c(1,2)))
names(querypoint) <- c('x', 'y')
querypoint

refpoints <- read.csv('./data/sample_ref_points.csv')
refpoints <- refpoints[,-1]
names(refpoints) <- c('A', 'x', 'y')
refpoints

rp3 <- c(refpoints[3,'x'], refpoints[3,'y'])
r3dist <- myDist(rp3, querypoint)

p1 <- point_from_angle(rp3[1], rp3[2], r3dist, 0)

data <- fade(rp3[1], rp3[2], r3dist-3, r3dist, 0, -pi/2.0, 20)

ref_lines = data.frame(x=refpoints[3, 'x'], y=refpoints[3,'y'], xend=data[,'x'], yend=data[,'y'])
ref_lines$dist <- sqrt((ref_lines$xend - ref_lines$x)^2 + (ref_lines$yend - ref_lines$y)^2)
ref_lines

ggplot() +
  geom_point(data=data, mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("One worst-case example for using single reference point R3")



