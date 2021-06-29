# https://medium.com/datascape/data-on-a-3d-world-map-cbe3702dbe67
library(rworldmap)
library(dplyr)
library(ggplot2)
library(geosphere)
library(gpclib)
library(rworldxtra)
dd <-data(countriesHigh)
# World map
worldMap <- getMap(resolution = "high")
world.points <- fortify(worldMap)
world.points$region <- world.points$idworld.df <- world.points[,c("long", "lat", "group")]
worldmap <- ggplot() + 
  geom_polygon(data = world.points, aes(x = long, y = lat, group = group)) +
  scale_y_continuous(breaks = (-2:2) * 30) +
  scale_x_continuous(breaks = (-4:4) * 45)
worldmap <- ggplot() + 
  geom_polygon(data = world.points, aes(x = long, y = lat, group = group), color="orange", fill="white") +
  scale_y_continuous(breaks = (-2:2) * 30) +
  scale_x_continuous(breaks = (-4:4) * 45) +
  coord_map("ortho", orientation=c(40, 90, 0))
worldmap
head(tail(world.points[world.points$region == "US",],10),4)


# https://www.rdocumentation.org/packages/swfscMisc/versions/1.5/topics/circle.polygon
cart.earth <- circle.polygon(-117.24, 32.86, 40, poly.type = "cart.earth")

lat.range <- c(32, 34)
lon.range <- c(-118.5, -116)

op <- par(mar = c(3, 5, 5, 5) + 0.1, oma = c(1, 1, 1, 1))

maps::map("mapdata::worldHires", fill = TRUE, col = "wheat3", xlim = lon.range, ylim = lat.range)
points(-117.24, 32.86, pch = 19, col = "red")
polygon(cart.earth, border = "red", lwd = 3)
lat.lon.axes(n = 3)
box(lwd = 2)
mtext("poly.type = 'cart.earth'", line = 3)

par(op)





# https://gis.stackexchange.com/questions/229453/create-a-circle-of-defined-radius-around-a-point-and-then-find-the-overlapping-a
library(raster)
library(tidyverse)
library(sf)
#> Linking to GEOS 3.5.0, GDAL 2.1.1, proj.4 4.9.3

# convert to sf format
dat_ticino_sf <- getData(name = "GADM",
                         country = "CHE",
                         level = 3) %>%
  # convert to simple features
  sf::st_as_sf() %>%
  # Filter down to Ticino
  dplyr::filter(NAME_1 == "Ticino") %>% 
  st_transform(3035) # Reproject to EPSG:3035 as mentioned above

mean_lat <- 46.07998
sd_lat <- 0.1609196
mean_long <- 8.931849
sd_long <-  0.1024659

set.seed(42)
dat_sim <- data.frame(lat = rnorm(500, mean = mean_lat, sd = sd_lat),
                      long = rnorm(500, mean = mean_long, sd = sd_long))

# Convert to sf, set the crs to EPSG:4326 (lat/long), 
# and transform to EPSG:3035
dat_sf <- st_as_sf(dat_sim, coords = c("long", "lat"), crs = 4326) %>% 
  st_transform(3035)

# Buffer circles by 100m
dat_circles <- st_buffer(dat_sf, dist = 100)

# Intersect the circles with the polygons
ticino_int_circles <- st_intersection(dat_ticino_sf, dat_circles)
#> Warning: attribute variables are assumed to be spatially constant
#> throughout all geometries

## Plot a zoomed in map of polygons and overlay intersected circles to double check:
## (I assumed NAME_3 is the best unique identifier for the polygons)
bb <- st_bbox(ticino_int_circles)

plot(dat_ticino_sf[, "NAME_3"], 
     xlim = c(mean(c(bb["xmin"], bb["xmax"])) - 1000, 
              mean(c(bb["xmin"], bb["xmax"])) + 1000), 
     ylim = c(mean(c(bb["ymin"], bb["ymax"])) - 1000, 
              mean(c(bb["ymin"], bb["ymax"])) + 1000))

plot(ticino_int_circles[, "NAME_3"], add = TRUE)

# Summarize by NAME_3
# which aggregates all of the circles by NAME_3, 
# then calculate the area
ticino_int_circles_summary <- group_by(ticino_int_circles, NAME_3) %>% 
  summarise() %>% 
  mutate(area = st_area(.))
