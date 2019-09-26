#
# Examples from: https://bwlewis.github.io/rthreejs/globejs.html
# 

library("threejs")
earth <- "http://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg"
globejs(img=earth, bg="white")

library("threejs")
library("maptools")
library("maps")
data(world.cities, package="maps")
cities <- world.cities[order(world.cities$pop,decreasing=TRUE)[1:1000],]
value  <- 100 * cities$pop / max(cities$pop)

globejs(bg="black", lat=cities$lat,     long=cities$long, value=value, 
        rotationlat=-0.34,     rotationlong=-0.38, fov=30)


# The three fixed Trilateration points
points <- data.frame(id = 1, lat = 90.000, long=0.000)  # The north pole
points <- rbind(points, c(2, 38.26, -85.76))  # Louisville, KY
points <- rbind(points, c(3, -9.42, 46.33))  # 
# points <- rbind(points, c(3, -19.22, 159.93))

arcs <- data.frame(cbind(points[c(1,2,3), c('lat', 'long')],
                         points[c(2,3,1), c('lat', 'long')]))

globejs(bg="blue", arcs = arcs,
        arcsHeight = 0.4, arcsLwd = 3, arcsOpacity = 0.7,
        arcsColor = 'white', img=earth)
