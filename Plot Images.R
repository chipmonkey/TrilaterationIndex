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
    
  }

  # 10 Random Points
  myPoints <- data.frame(x = runif(n = 10, min = 0, max = 100),
                   y = runif(n = 10, min = 0, max = 100))
  
  myTrilat <- data.frame(d1=NULL, d2=NULL, d3=NULL)
  for(i in 1:nrow(myPoints)) {
    myTrilat[i, 'd1'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[1,'x'], myTri[1,'x']))
    myTrilat[i, 'd2'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[2,'x'], myTri[2,'x']))
    myTrilat[i, 'd3'] <- myDist(c(myPoints[i,'x'], myPoints[i, 'y']), c(myTri[3,'x'], myTri[3,'x']))
  }
   
  myTrilateration <- cbind(myPoints, myTrilat)
  myTrilateration
  
  plot(x=NULL, y=NULL, xlim=c(0,100), ylim=c(0,100), asp = 1,
       xlab='X', ylab='Y')
  points(myTrilateration$x, myTrilateration$y)
  draw.circle(x = 50, y = 0, radius = 1, lwd = 1, col="red")
  draw.circle(x = 25*(2+sqrt(3)), y = tan(pi/3)*((25*(2+sqrt(3)))-50), radius = 1, lwd = 1, col="red")
  draw.circle(x = -25*(sqrt(3)-2), y = tan(-pi/3)*((-25*(sqrt(3)-2))-50), radius = 1, lwd = 1, col="red")
  