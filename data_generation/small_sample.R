# File paths are relative to this file's location, so if needed set pwd:
# setwd(getSrcDirectory()[1])

# Distance function which I think we don't use:
myDist <- function(x, y) {
  # x and y must be 2D points -- (x,y) tuples i.e.: x <- c(1,2) y <- c(3,4)
  d <- sqrt((x[[1]]-y[[1]])^2+(x[[2]]-y[[2]])^2)
  return(d)
}

# Function to generate lots of random points
makepoints <- function(n, min, max, myseed=1729) {
  set.seed(myseed)
  myPoints <- data.frame(x = runif(n = n, min = min, max = max),
                         y = runif(n = n, min = min, max = max))
  return(myPoints)
}


# 10 Random Points (seed ensures reproducibility):
myPoints <- makepoints(10, 0, 100)
myPoints
write.csv(myPoints, '../data/point_sample_10.csv')

# 10 Other random points for Q (make 20 to avoid random seed collision)
qpoints <- makepoints(20, 0, 100)
qpoints[11:20,]
write.csv(qpoints[11:20,], '../data/query_sample_10.csv')

# And a Point and Query sample for 100 points:
write.csv(makepoints(100, 0, 100, myseed=1975), '../data/point_sample_100.csv')
write.csv(makepoints(100, 0, 100, myseed=1776), '../data/query_sample_100.csv')
