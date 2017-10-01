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

  # 10 Random Points (seed ensures reproducibility):
  set.seed(1729)
  myPoints <- data.frame(x = runif(n = 10, min = 0, max = 100),
                    y = runif(n = 10, min = 0, max = 100))
  myPoints

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
  

# Plotting things...
  
  plot(myTrilateration[,1])
  lines(myTrilateration[,1])
  points(myTrilateration[,2], col=2)
  lines(myTrilateration[,2], col=2)
  points(myTrilateration[,3], col=3)
  lines(myTrilateration[,3], col=3)
  points(myTrilateration[,4], col=4)
  lines(myTrilateration[,4], col=4)
  points(myTrilateration[,5], col=5)
  lines(myTrilateration[,5], col=5)
  