library(ggplot2)

setwd("~/repos/TrilaterationIndex")

myDist <- function(x, y) {
  # x and y must be 2D points -- (x,y) tuples i.e.: x <- c(1,2) y <- c(3,4)
  d <- sqrt((x[[1]]-y[[1]])^2+(x[[2]]-y[[2]])^2)
  return(d)
}

data <- read.csv('./data/point_sample_100.csv')
data <- data[,-1]
data

refpoints <- read.csv('./data/sample_ref_points.csv')
refpoints <- refpoints[,-1]
names(refpoints) <- c('A', 'x', 'y')
refpoints

querypoint <- data.frame(array(c(50, 65), dim=c(1,2)))
names(querypoint) <- c('x', 'y')
querypoint

ggplot() +
  geom_point(data=data, mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(0, 100))


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
  ggtitle("Distances from R3")

r3dist = apply(ref_lines, 1, function(x) sqrt((x[3]-x[1])^2 + (x[4]-x[2])^2))
data$r3dist <- r3dist
data <- data[order(data$r3dist),]
data

r1dist = apply(data, 1, function(x) sqrt((x[1] - refpoints[1,'x'])^2 + (x[2] - refpoints[1, 'y'])^2))
data$r1dist <- r1dist

r2dist = apply(data, 1, function(x) sqrt((x[1] - refpoints[2,'x'])^2 + (x[2] - refpoints[2, 'y'])^2))
data$r2dist <- r2dist

data2 <- data[order(data$r2dist),]

qdists = apply(refpoints, 1, function(x) sqrt((querypoint$x - x[2])^2 + (querypoint$y - x[3])^2))
qdists <- data.frame(array(qdists, dim=c(1,3)))
names(qdists) <- c('r1dist', 'r2dist', 'r3dist')
querypoint <- cbind(querypoint, qdists)
querypoint


#---------------------

ggplot() +
  geom_point(data=data, mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==74,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  geom_point(data=data[rownames(data)==64,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  
  geom_segment(data=ref_lines[ref_lines$dist >= 40 & ref_lines$dist <= 50,], mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("P74 is the nearest point WRT distance to R3")

data[data$r3dist >= 40 & data$r3dist <= 50, ]
data2 <- data


#------------------------
qp74dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==74, 'x'], data[rownames(data)==74, 'y']))
library(ggforce)
ggplot() +
  geom_point(data=data, mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==74,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
#  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp74dist), data=querypoint, linetype="dashed") +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("Only four points remain")


#------------------------
data.orig = data
data <- data[abs(data$r3dist-querypoint$r3dist) <= qp74dist,]
nrow(data)

ref_lines = data.frame(x=refpoints[3, 'x'], y=refpoints[3,'y'], xend=data[,'x'], yend=data[,'y'])
ref_lines$dist <- sqrt((ref_lines$xend - ref_lines$x)^2 + (ref_lines$yend - ref_lines$y)^2)
ref_lines

nexti = data[which.min(abs(data[row.names(data) != 74,]$r3dist - querypoint$r3dist)),]
x = cbind(data$x, data$y, abs(data$r3dist-querypoint$r3dist),
          abs(data$r2dist - querypoint$r2dist),
          abs(data$r1dist - querypoint$r1dist))
row.names(x) <- row.names(data)
x

# Skip to 50 since all other points can be "no closer than" current best distance

qp50dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==50, 'x'],
                                                    data[rownames(data)==50, 'y']))


library(ggforce)
ggplot() +
  geom_arc_bar(data=refpoints[3,],
               aes(x0=x,y0=y,
                   r0=querypoint$r3dist - qp50dist,
                   r=querypoint$r3dist + qp50dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE)) +
  geom_point(data=data[rownames(data) %in% c(1,7,4,6),], mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=data[!rownames(data) %in% c(1,7,4,6),], mapping=aes(x=x, y=y), color="black", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==50,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  #  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp50dist), data=querypoint, linetype="dashed") +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("Only four points remain") +
  theme(legend.position = "none")

#------------------------

data <- data[abs(data$r3dist-querypoint$r3dist) <= qp50dist &
             abs(data$r2dist-querypoint$r2dist) <= qp50dist &
             abs(data$r1dist-querypoint$r1dist) <= qp50dist,]
nrow(data)
data = data[!row.names(data) %in% c(50, 74),]

ref_lines = data.frame(x=refpoints[3, 'x'], y=refpoints[3,'y'], xend=data[,'x'], yend=data[,'y'])
ref_lines$dist <- sqrt((ref_lines$xend - ref_lines$x)^2 + (ref_lines$yend - ref_lines$y)^2)
ref_lines

nexti = data[which.min(abs(data$r3dist - querypoint$r3dist)),]


qp93dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==93, 'x'], data[rownames(data)==93, 'y']))
library(ggforce)
ggplot() +
  geom_arc_bar(data=refpoints[3,],
               aes(x0=x,y0=y,
                   r0=querypoint$r3dist - qp50dist,
                   r=querypoint$r3dist + qp50dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[2,],
               aes(x0=x,y0=y,
                   r0=querypoint$r2dist - qp50dist,
                   r=querypoint$r2dist + qp50dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[1,],
               aes(x0=x,y0=y,
                   r0=querypoint$r1dist - qp50dist,
                   r=querypoint$r1dist + qp50dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_point(data=data[rownames(data) %in% c(1,7,4,6),], mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=data[!rownames(data) %in% c(1,7,4,6),], mapping=aes(x=x, y=y), color="black", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==93,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  #  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp2dist), data=querypoint, linetype="dashed") +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("If needed we could purge points using the other reference points") +
  theme(legend.position = "none")


