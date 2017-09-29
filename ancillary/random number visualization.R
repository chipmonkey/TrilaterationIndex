# Code to apply the "Pickover Test" to the random number generators for various probability distributions
# http://web.uvic.ca/~dgiles/blog/Pickover.R
# https://davegiles.blogspot.ca/p/code.html

library("rgl")

# r<- rnorm(99000)
#r<- rbinom(99000,100,0.3)
#r<- rpois(99000,10)
#r<- runif(9000)
r<- rt(99000, 3)  # Try different degrees of freedom
#r<- rlogis(99000)
#r<- rbeta(99000, 1,2)
#r<- rchisq(99000,15)  # Try different degrees of freedom

x <- r[seq(1, length(r), 3)]
y<- r[seq(2, length(r), 3)]
z<- r[seq(3, length(r), 3)]

lim <- function(x){c(-max(abs(x)), max(abs(x))) * 1.1}

rgl.open()
rgl.bg(color="white")
rgl.spheres(x, y, z, r = 0.5, color = "lightblue")
rgl.lines(lim(x), c(0, 0), c(0, 0), color = "black", lw=1.5)
rgl.lines(c(0, 0), lim(y), c(0, 0), color = "red", lw=1.5)
rgl.lines(c(0, 0), c(0, 0), lim(z), color = "blue", lw=1.5)
grid3d(c("x", "y", "z"))