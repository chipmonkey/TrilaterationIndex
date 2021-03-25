# Estimate area of arc
setwd("~/repos/TrilaterationIndex/RScripts")
library(ggplot2)
library(ggforce)

myDist <- function(x, y) {
  # x and y must be 2D points -- (x,y) tuples i.e.: x <- c(1,2) y <- c(3,4)
  d <- sqrt((x[[1]]-y[[1]])^2+(x[[2]]-y[[2]])^2)
  return(d)
}


data <- read.csv('../data/point_sample_10.csv')
data <- data[,-1]

refpoints <- read.csv('../data/sample_ref_points.csv')
refpoints <- refpoints[c(3,2,1),-1]
row.names(refpoints) <- c(1,2,3)
names(refpoints) <- c('A', 'x', 'y')
refpoints$A <- c(1,2,3)

querypoint <- data.frame(array(c(50, 65), dim=c(1,2)))
names(querypoint) <- c('x', 'y')

ref_lines = data.frame(x=refpoints[1, 'x'], y=refpoints[1,'y'], xend=data[,'x'], yend=data[,'y'])

r3dist = apply(ref_lines, 1, function(x) sqrt((x[3]-x[1])^2 + (x[4]-x[2])^2))
data$r3dist <- r3dist
data <- data[order(data$r3dist),]

r1dist = apply(data, 1, function(x) sqrt((x[1] - refpoints[1,'x'])^2 + (x[2] - refpoints[1, 'y'])^2))
data$r1dist <- r1dist

r2dist = apply(data, 1, function(x) sqrt((x[1] - refpoints[2,'x'])^2 + (x[2] - refpoints[2, 'y'])^2))
data$r2dist <- r2dist

qdists = apply(refpoints, 1, function(x) sqrt((querypoint$x - x[2])^2 + (querypoint$y - x[3])^2))
qdists <- data.frame(array(qdists, dim=c(1,3)))
names(qdists) <- c('r1dist', 'r2dist', 'r3dist')
querypoint <- cbind(querypoint, qdists)

qp2dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==2, 'x'], data[rownames(data)==2, 'y']))
ggplot() +
  geom_arc_bar(data=refpoints[1,],
               aes(x0=x,y0=y,
                   r0=querypoint$r1dist - qp2dist,
                   r=querypoint$r1dist + qp2dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE)) +
  geom_point(data=data[rownames(data) %in% c(1,7,4,6),], mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=data[!rownames(data) %in% c(1,7,4,6),], mapping=aes(x=x, y=y), color="black", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==2,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  #  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp2dist), data=querypoint, linetype="dashed") +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  theme(legend.position = "none")


test <- data.frame(x = runif(10000, min=0, max=100), y=runif(1000, min=0, max=100))
# ggplot() +
#   geom_arc_bar(data=refpoints[1,],
#                aes(x0=x,y0=y,
#                    r0=querypoint$r1dist - qp2dist,
#                    r=querypoint$r1dist + qp2dist,
#                    start=-pi/2, end=1.5*pi, fill=TRUE)) +
#   geom_point(data=test, mapping=aes(x=x, y=y), color="black", size=1) +
#   geom_point(data=refpoints[1,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
#   coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
#   theme(legend.position = "none")

tdists <- apply(test, 1, function(x) sqrt((refpoints[1,'x'] - x[1])^2 + (refpoints[1, 'y'] - x[2])^2))
lowbound = qdists$r1dist - qp2dist
highbound = qdists$r1dist + qp2dist

tin <- test[which(tdists >= lowbound & tdists <= highbound),]
tout <- test[which(tdists < lowbound | tdists > highbound),]

ratio = nrow(tin) / nrow(test)
ratio

# ggplot() +
#   geom_arc_bar(data=refpoints[1,],
#                aes(x0=x,y0=y,
#                    r0=querypoint$r1dist - qp2dist,
#                    r=querypoint$r1dist + qp2dist,
#                    start=-pi/2, end=1.5*pi, fill=TRUE)) +
#   geom_point(data=tin, mapping=aes(x=x, y=y), color="green", size=1) +
#   geom_point(data=tout, mapping=aes(x=x, y=y), color="black", size=1) +
#   geom_point(data=refpoints[1,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
#   coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
#   theme(legend.position = "none")


#  Plots with qp6dist and multiple r:

qp6dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==6, 'x'], data[rownames(data)==6, 'y']))

t1d <- apply(test, 1, function(x) sqrt((refpoints[1,'x'] - x[1])^2 + (refpoints[1, 'y'] - x[2])^2))
t2d <- apply(test, 1, function(x) sqrt((refpoints[2,'x'] - x[1])^2 + (refpoints[2, 'y'] - x[2])^2))
t3d <- apply(test, 1, function(x) sqrt((refpoints[3,'x'] - x[1])^2 + (refpoints[3, 'y'] - x[2])^2))

low1 = qdists$r1dist - qp6dist
high1 = qdists$r1dist + qp6dist
low2 = qdists$r2dist - qp6dist
high2 = qdists$r2dist + qp6dist
low3 = qdists$r3dist - qp6dist
high3 = qdists$r3dist + qp6dist

r1in <- test[which(t1d >= low1 & t1d <= high1),]
r2in <- test[which(t2d >= low2 & t2d <= high2),]
r3in <- test[which(t3d >= low3 & t3d <= high3),]
tin <- test[which(t1d >= low1 & t1d <= high1 & t2d >= low2 & t2d <= high2 & t3d >= low3 & t3d <= high3),]
tout <- test[which(t1d < low1 | t1d > high1 | t2d < low2 | t2d > high2 | t3d < low3 | t3d > high3),]

ggplot() +
  geom_point(data=tout, mapping=aes(x=x, y=y), color="black", size=1) +
  geom_arc_bar(data=refpoints[3,],
               aes(x0=x,y0=y,
                   r0=querypoint$r3dist - qp6dist,
                   r=querypoint$r3dist + qp6dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[2,],
               aes(x0=x,y0=y,
                   r0=querypoint$r2dist - qp6dist,
                   r=querypoint$r2dist + qp6dist,
                   start=pi/2, end=-1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[1,],
               aes(x0=x,y0=y,
                   r0=querypoint$r1dist - qp6dist,
                   r=querypoint$r1dist + qp6dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_point(data=refpoints[1:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_circle(aes(x0=x, y0=y, r=qp6dist), data=querypoint, linetype="dashed") +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=tin, mapping=aes(x=x, y=y), color="green", size=1) +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  theme(legend.position = "none")

ratio = nrow(tin) / nrow(test)
ratio
r1ratio = nrow(r1in) / nrow(test)
r1ratio
r2ratio = nrow(r2in) / nrow(test)
r2ratio
r3ratio = nrow(r3in) / nrow(test)
r3ratio
