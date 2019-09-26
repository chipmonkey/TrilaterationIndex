#
# Examples from: http://eriqande.github.io/rep-res-web/lectures/making-maps-with-R.html
# 


library('maps')
library("mapproj")
m = map("state", fill = TRUE, plot = FALSE)
area.map(m, "Kentucky")

library("ggplot2")
usa <- map_data("usa") 
states <- map_data("state")
ggplot() + 
  geom_polygon(data = usa, aes(x=long, y = lat, group = group), fill = NA, color = "red") + 
  coord_fixed(1.3)

ggplot(data = states) + 
  geom_polygon(aes(x = long, y = lat, fill = region, group = group), color = "white") + 
  coord_fixed(1.3) +
  guides(fill=FALSE)  # do this to leave off the color legend

kentucky <- subset(states, region %in% "kentucky")


sampleLL <- read.csv('data/lat_long_synthetic.csv')

# Data is all over the country -- make a simple bounding box for Kentucky:
  minlong <- min(kentucky$long)
  minlat <- min(kentucky$lat)
  maxlong <- max(kentucky$long)
  maxlat <- max(kentucky$lat)

  x <- which (sampleLL$Longitude > minlong &
                     sampleLL$Longitude < maxlong &
                     sampleLL$Latitude > minlat &
                     sampleLL$Latitude < maxlat)
sampleLL <- sampleLL[x,]

counties <- map_data('county')
ky_counties <- subset(counties, region == "kentucky")

ggplot(data = kentucky) + 
  geom_polygon(aes(x = long, y = lat, group = group), fill = "deepskyblue", color = "black") + 
  coord_fixed(1.3) +
  geom_point(data=sampleLL, aes(x=Longitude, y=Latitude), alpha = 0.1) +
  geom_polygon(data=ky_counties, aes(x=long, y=lat, group=group), fill = NA, color = "white")
  

