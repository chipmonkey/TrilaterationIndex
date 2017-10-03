rm(list=ls()) ; gc()
library(plotrix)

plot(x=NULL, y=NULL, xlim=c(0,100), ylim=c(0,100), asp = 1,
     xlab='X', ylab='Y')
draw.circle(x = 50, y = 50, radius = 50)
abline(h=0)
abline(h=100)
abline(v=0)
abline(v=100)

# Draw point
draw.circle(x = 50, y = 0, radius = 1, lwd = 1, col="red")

# Triangle points:
myTri <- data.frame(id=1, x=50, y=0)
myTri <- rbind(myTri, c(2, 25*(2+sqrt(3)), tan(pi/3)*((25*(2+sqrt(3)))-50)))
myTri <- rbind(myTri, c(3, -25*(sqrt(3)-2), tan(-pi/3)*((-25*(sqrt(3)-2))-50)))
polygon(myTri[,2], myTri[,3], border="blue")

# abline(a=tan(pi/3)*(-50), tan(pi/3))


# Draw point
draw.circle(x = 50, y = 0, radius = 1, lwd = 1, col="red")
draw.circle(x = 25*(2+sqrt(3)), y = tan(pi/3)*((25*(2+sqrt(3)))-50), radius = 1, lwd = 1, col="red")
draw.circle(x = -25*(sqrt(3)-2), y = tan(-pi/3)*((-25*(sqrt(3)-2))-50), radius = 1, lwd = 1, col="red")


# 2D Examples:
  # Distance function:
  myDist <- function(x, y) {
    # x and y must be 2D points -- (x,y) tuples i.e.: x <- c(1,2) y <- c(3,4)
    d <- sqrt((x[[1]]-y[[1]])^2+(x[[2]]-y[[2]])^2)
    return(d)
  }

  # Function to generate lots of random points
  makepoints <- function(n, min, max) {
    set.seed(1729)
    myPoints <- data.frame(x = runif(n = n, min = min, max = max),
                           y = runif(n = n, min = min, max = max))
    return(myPoints)
  }

  # 10 Random Points (seed ensures reproducibility):
  myPoints <- makepoints(10, 0, 100)

  #    x        y
  # 1  58.52396 53.516719
  # 2  43.73940 43.418684
  # 3  57.28944  7.950161
  # 4  35.32139 58.321179
  # 5  86.12714 52.201894
  # 6  41.08036 78.065907
  # 7  51.14533 47.157734
  # 8  15.42852 80.836340
  # 9  85.13531 64.090063
  # 10 99.60833 78.055071
  
  myTrilat <- data.frame(d1=NULL, d2=NULL, d3=NULL)
  for(i in 1:nrow(myPoints)) {
    myTrilat[i, 'd1'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[1,'x'], myTri[1,'y']))
    myTrilat[i, 'd2'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[2,'x'], myTri[2,'y']))
    myTrilat[i, 'd3'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[3,'x'], myTri[3,'y']))
  }
   
  myTrilateration <- cbind(myPoints, myTrilat)
  myTrilateration
  
  rm(myTrilat)  #twas temporary
  
  plot(x=NULL, y=NULL, xlim=c(0,100), ylim=c(0,100), asp = 1,
       xlab='X', ylab='Y')
  points(myTrilateration$x, myTrilateration$y, col="blue", pch=16, asp=1)
  
  draw.circle(x = myTri[1,'x'], y = myTri[1,'y'], radius = 1, lwd = 1, col="red")
  draw.circle(x = myTri[2,'x'], y = myTri[2,'y'], radius = 1, lwd = 1, col="red")
  draw.circle(x = myTri[3,'x'], y = myTri[3,'y'], radius = 1, lwd = 1, col="red")


# Arcs around two points:
  
  plot(x=NULL, y=NULL, xlim=c(0,100), ylim=c(0,100), asp = 1,
       xlab='X', ylab='Y')
  points(myTrilateration[c(2,10),'x'], myTrilateration[c(2,10),'y'], col="blue", pch=16, asp=1)
  draw.circle(x = myTri[1,'x'], y = myTri[1,'y'], radius = 1, lwd = 1, col="red")
  draw.circle(x = myTri[2,'x'], y = myTri[2,'y'], radius = 1, lwd = 1, col="red")
  draw.circle(x = myTri[3,'x'], y = myTri[3,'y'], radius = 1, lwd = 1, col="red")
  
  draw.circle(x=myTri[1,'x'], y=myTri[1,'y'], radius=myTrilateration[2,'d1'])
  draw.circle(x=myTri[1,'x'], y=myTri[1,'y'], radius=myTrilateration[10,'d1'])

  draw.circle(x=myTri[2,'x'], y=myTri[2,'y'], radius=myTrilateration[2,'d2'])
  draw.circle(x=myTri[2,'x'], y=myTri[2,'y'], radius=myTrilateration[10,'d2'])

  draw.circle(x=myTri[3,'x'], y=myTri[3,'y'], radius=myTrilateration[2,'d3'])
  draw.circle(x=myTri[3,'x'], y=myTri[3,'y'], radius=myTrilateration[10,'d3'])
  

# Triangles...
  plot(x=NULL, y=NULL, xlim=c(0,100), ylim=c(0,100), asp = 1,
       xlab='X', ylab='Y')
  
  # reference points:
    draw.circle(x = myTri[1,'x'], y = myTri[1,'y'], radius = 1, lwd = 1, col="red")
    draw.circle(x = myTri[2,'x'], y = myTri[2,'y'], radius = 1, lwd = 1, col="red")
    draw.circle(x = myTri[3,'x'], y = myTri[3,'y'], radius = 1, lwd = 1, col="red")
    
    text(myTri[1, 'x'], myTri[1, 'y'], labels = 'P1', pos=2)
    text(myTri[2, 'x'], myTri[2, 'y'], labels = 'P2', pos=4)
    text(myTri[3, 'x'], myTri[3, 'y'], labels = 'P3', pos=2)
  # Target points
    points(myTrilateration[c(1,4),'x'], myTrilateration[c(1,4),'y'], col="blue", pch=16, asp=1)
    text(myTrilateration[1,'x'], myTrilateration[1,'y'], labels = 'x', pos=4)
    text(myTrilateration[4,'x'], myTrilateration[4,'y'], labels = 'y', pos=2)
  # Target distance
    lines(myTrilateration[c(1,4),'x'], myTrilateration[c(1,4),'y'], col="red")
  
  # Trilateration 1
    lines(c(myTri[1, 'x'], myTrilateration[1,'x']), c(myTri[1, 'y'], myTrilateration[1, 'y']), col="green")
    lines(c(myTri[2, 'x'], myTrilateration[1,'x']), c(myTri[2, 'y'], myTrilateration[1, 'y']), col="green")
    lines(c(myTri[3, 'x'], myTrilateration[1,'x']), c(myTri[3, 'y'], myTrilateration[1, 'y']), col="green")

  # Trilateration 2
    lines(c(myTri[1, 'x'], myTrilateration[4,'x']), c(myTri[1, 'y'], myTrilateration[4, 'y']), col="blue")
    # lines(c(myTri[2, 'x'], myTrilateration[4,'x']), c(myTri[2, 'y'], myTrilateration[4, 'y']), col="blue")
    lines(c(myTri[3, 'x'], myTrilateration[4,'x']), c(myTri[3, 'y'], myTrilateration[4, 'y']), col="blue")
  
  # Reference Equilateral Triangle
    lines(myTri[c(1,2,3,1), 'x'], myTri[c(1,2,3,1), 'y'])
    
rDistP1P2 <- myDist(myTri[1,c('x','y')], myTri[2, c('x', 'y')]) # It's an Equilateral triangle
rDistP2P3 <- myDist(myTri[2,c('x','y')], myTri[3, c('x', 'y')]) # So these should be the same
rDistP3P1 <- myDist(myTri[3,c('x','y')], myTri[1, c('x', 'y')]) # But let's do this in case we change things
  

# Heron's Formula: http://www.mathopenref.com/heronsformula.html
p_P1yP3 <- (rDistP3P1 + myTrilateration[4, 'd1'] + myTrilateration[4, 'd3']) / 2.0
areaP1yP3 <- sqrt(p_P1yP3*(p_P1yP3 - rDistP3P1)*(p_P1yP3 - myTrilateration[4,'d1'])*(p_P1yP3 - myTrilateration[4, 'd3']))

p_P2xP3 <- (rDistP2P3 + myTrilateration[1, 'd2'] + myTrilateration[1, 'd3']) / 2.0
p <- p_P2xP3
areaP2xP3 <- sqrt(p*(p - rDistP2P3) * (p - myTrilateration[1, 'd2']) * (p - myTrilateration[1, 'd3']))

p_P1xP2 <- (rDistP1P2 + myTrilateration[1, 'd1'] + myTrilateration[1, 'd2']) / 2.0
p <- p_P1xP2
areaP1xP2 <- sqrt(p*(p - rDistP1P2) * (p - myTrilateration[1, 'd1']) * (p - myTrilateration[1, 'd2']))

areaP1yP3
areaP2xP3
areaP1xP2

# Equilateral triangle for total area:
areaP1P2P3 <- (sqrt(3)/4)*rDistP1P2^2

areaP1P2P3

# Remaining area for P3xy and P1xy:
myArea <- areaP1P2P3 - (areaP1yP3 + areaP2xP3 + areaP1xP2)

# So: Area P3xy + Area P1xy = myArea
# p_a <- (d(p3,x) + d(p3, y) + d(x,y))/2
# p_b <- (d(p1,x) + d(p1, y) + d(x,y))/2
# a = sqrt(p_a*(p_a-d(p3, x))*(p_a-d(p3,y))*(p_a-d(x,y)))
# b = sqrt(p_b*(p_b-d(p1, x))*(p_b-d(p1,y))*(p_b-d(x,y)))
# a + b = myArea


# Plotting things...
  
  plot(myTrilateration[,1], ylim=c(0,100))
  lines(myTrilateration[,1])
  points(myTrilateration[,2], col=2)
  lines(myTrilateration[,2], col=2)
  points(myTrilateration[,3], col=3)
  lines(myTrilateration[,3], col=3)
  points(myTrilateration[,4], col=4)
  lines(myTrilateration[,4], col=4)
  points(myTrilateration[,5], col=5)
  lines(myTrilateration[,5], col=5)
  
  
manyPoints <- makepoints(10000, 0, 100)

maketrilat <- function(myPoints, myTri) {
  myTrilat <- data.frame(d1=NULL, d2=NULL, d3=NULL)
  for(i in 1:nrow(myPoints)) {
    myTrilat[i, 'd1'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[1,'x'], myTri[1,'y']))
    myTrilat[i, 'd2'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[2,'x'], myTri[2,'y']))
    myTrilat[i, 'd3'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[3,'x'], myTri[3,'y']))
  }
  myTrilateration <- cbind(myPoints, myTrilat)
  return(myTrilateration)
}


myTrilateration <- maketrilat(manyPoints, myTri)
head(myTrilateration)

plot(myTrilateration[,1], ylim=c(0,100), col=rgb(1,0,0,0.2))
# lines(myTrilateration[,1])
points(myTrilateration[,2], col=rgb(0,1,0,0.2))
lines(myTrilateration[,2], col=rgb(0,1,0,0.2))
points(myTrilateration[,3], col=rgb(0,0,1,0.2))
lines(myTrilateration[,3], col=rgb(0,0,1,0.2))
points(myTrilateration[,4], col=rgb(0.5,0.5,0,0.2))
lines(myTrilateration[,4], col=rgb(0.5,0.5,0,0.2))
points(myTrilateration[,5], col=rgb(0,0.5,0.5,0.2))
lines(myTrilateration[,5], col=rgb(0,0.5,0.5,0.2))


