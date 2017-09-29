library(plotrix)

plot(x=NULL, y=NULL, xlim=c(0,100), ylim=c(0,100), asp = 1)
draw.circle(x = 50, y = 50, radius = 50)
abline(h=0)
abline(h=100)
abline(v=0)
abline(v=100)

# Triangle points:
myTri <- data.frame(id=1, x=50, y=0)
myTri <- rbind(myTri, c(2, 25*(2+sqrt(3)), tan(pi/6)*((25*(2+sqrt(3)))-50)))
myTri <- rbind(myTri, c(3, -25*(sqrt(3)-2), tan(-pi/6)*((-25*(sqrt(3)-2))-50)))
polygon(myTri[,2], myTri[,3])

abline(a=tan(pi/6)*(-50), tan(pi/6))
