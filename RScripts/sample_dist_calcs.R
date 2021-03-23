library(ggplot2)

setwd("~/repos/TrilaterationIndex")

data <- read.csv('./data/point_sample_10.csv')
data <- data[,-1]
data

refpoints <- read.csv('./data/sample_ref_points.csv')
refpoints <- refpoints[c(3,2,1),-1]
row.names(refpoints) <- c(1,2,3)
names(refpoints) <- c('A', 'x', 'y')
refpoints$A <- c(1,2,3)
refpoints

querypoint <- data.frame(array(c(50, 65), dim=c(1,2)))
names(querypoint) <- c('x', 'y')
querypoint

ggplot() +
  geom_point(data=data, mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(0, 100))


ref_lines = data.frame(x=refpoints[1, 'x'], y=refpoints[1,'y'], xend=data[,'x'], yend=data[,'y'])
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

data

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
  geom_point(data=data[rownames(data)==2,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("P2 is the nearest point WRT distance to R3")

data[rownames(data) %in% c(2,4,7), ]

myDist <- function(x, y) {
  # x and y must be 2D points -- (x,y) tuples i.e.: x <- c(1,2) y <- c(3,4)
  d <- sqrt((x[[1]]-y[[1]])^2+(x[[2]]-y[[2]])^2)
  return(d)
}


#------------------------
qp2dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==2, 'x'], data[rownames(data)==2, 'y']))
library(ggforce)
ggplot() +
  geom_point(data=data, mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==2,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
#  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp2dist), data=querypoint, linetype="dashed") +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("Only four points remain")


#------------------------

qp2dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==2, 'x'], data[rownames(data)==2, 'y']))
library(ggforce)
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
  ggtitle("Only four points remain") +
  theme(legend.position = "none")

#------------------------
qp2dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==2, 'x'], data[rownames(data)==2, 'y']))
library(ggforce)
ggplot() +
  geom_arc_bar(data=refpoints[3,],
               aes(x0=x,y0=y,
                   r0=querypoint$r3dist - qp2dist,
                   r=querypoint$r3dist + qp2dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[2,],
               aes(x0=x,y0=y,
                   r0=querypoint$r2dist - qp2dist,
                   r=querypoint$r2dist + qp2dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[1,],
               aes(x0=x,y0=y,
                   r0=querypoint$r1dist - qp2dist,
                   r=querypoint$r1dist + qp2dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
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
  ggtitle("If needed we could purge points using the other reference points") +
  theme(legend.position = "none")


#-----------------------

data2 = data[rownames(data) %in% c(1,2,4,7,6),]

# p7 is the next closest point; see that querypoint$r3dist (44.44) is
# closer to data$r3dist for p7 (52.44) than p6 (34.52)

qp7dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==7, 'x'], data[rownames(data)==7, 'y']))

ggplot() +
  geom_arc_bar(data=refpoints[3,],
               aes(x0=x,y0=y,
                   r0=querypoint$r3dist - qp7dist,
                   r=querypoint$r3dist + qp7dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE)) +
  geom_arc_bar(data=refpoints[2,],
               aes(x0=x,y0=y,
                   r0=querypoint$r2dist - qp7dist,
                   r=querypoint$r2dist + qp7dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[1,],
               aes(x0=x,y0=y,
                   r0=querypoint$r1dist - qp7dist,
                   r=querypoint$r1dist + qp7dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_point(data=data[rownames(data) %in% c(1,4,6),], mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=data[!rownames(data) %in% c(1,4,6),], mapping=aes(x=x, y=y), color="black", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -5, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -5) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==7,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  #  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp7dist), data=querypoint, linetype="dashed") +
  
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("Only three points remain") +
  theme(legend.position = "none")


#-----------------------

# p7 is the next closest point; see that querypoint$r3dist (44.44) is
# closer to data$r3dist for p7 (52.44) than p6 (34.52)

qp6dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==6, 'x'], data[rownames(data)==6, 'y']))

ggplot() +
  geom_arc_bar(data=refpoints[3,],
               aes(x0=x,y0=y,
                   r0=querypoint$r3dist - qp6dist,
                   r=querypoint$r3dist + qp6dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE)) +
  geom_arc_bar(data=refpoints[2,],
               aes(x0=x,y0=y,
                   r0=querypoint$r2dist - qp6dist,
                   r=querypoint$r2dist + qp6dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_arc_bar(data=refpoints[1,],
               aes(x0=x,y0=y,
                   r0=querypoint$r1dist - qp6dist,
                   r=querypoint$r1dist + qp6dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_point(data=data[rownames(data) %in% c(1,4),], mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=data[!rownames(data) %in% c(1,4),], mapping=aes(x=x, y=y), color="black", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -2, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -2) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==6,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  #  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp6dist), data=querypoint, linetype="dashed") +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1, xlim=c(0,120), ylim=c(-5, 100)) +
  ggtitle("Only two points remain") +
  theme(legend.position = "none")


ggplot() +
  geom_arc_bar(data=refpoints[3,],
               aes(x0=x,y0=y,
                   r0=querypoint$r3dist - qp6dist,
                   r=querypoint$r3dist + qp6dist,
                   start=-pi/2, end=1.5*pi, fill=TRUE)) +
  # geom_arc_bar(data=refpoints[2,],
  #              aes(x0=x,y0=y,
  #                  r0=querypoint$r2dist - qp6dist,
  #                  r=querypoint$r2dist + qp6dist,
  #                  start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  # geom_arc_bar(data=refpoints[1,],
  #              aes(x0=x,y0=y,
  #                  r0=querypoint$r1dist - qp6dist,
  #                  r=querypoint$r1dist + qp6dist,
  #                  start=-pi/2, end=1.5*pi, fill=TRUE, alpha=0.5)) +
  geom_point(data=data[rownames(data) %in% c(1,4),], mapping=aes(x=x, y=y), color="blue", size=2) +
  geom_point(data=data[!rownames(data) %in% c(1,4),], mapping=aes(x=x, y=y), color="black", size=2) +
  geom_text(data=data, aes(x=x, y=y, label=rownames(data)), nudge_y = -2, color="blue") +
  geom_point(data=refpoints[0:3,], mapping=aes(x=x, y=y), color="red", shape="triangle", size=3) +
  geom_text(data=refpoints[0:3,], aes(x=x, y=y, label=A), nudge_y = -2) +
  geom_point(data=querypoint, mapping=aes(x=x,y=y), color="green4", shape="square", size=3) +
  geom_point(data=data[rownames(data)==6,], aes(x=x, y=y), color="orange", shape="diamond", size=4) +
  #  geom_point(data=data[rownames(data) %in% c(4, 7),], aes(x=x, y=y), color="pink", size=4) +
  geom_circle(aes(x0=x, y0=y, r=qp6dist), data=querypoint, linetype="dashed") +
  geom_segment(data=ref_lines, mapping=aes(x=x, y=y, xend=xend, yend=yend), linetype="dotted") +
  coord_fixed(ratio = 1,
              xlim=c(querypoint$x - qp2dist, querypoint$x + qp2dist),
              ylim=c(querypoint$y - qp2dist, querypoint$y + qp2dist)) +
  ggtitle("Zoom - Only two points remain") +
  theme(legend.position = "none")


qp4dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==4, 'x'], data[rownames(data)==4, 'y']))
qp1dist <- myDist(c(querypoint$x, querypoint$y), c(data[rownames(data)==1, 'x'], data[rownames(data)==1, 'y']))
qp1dist

