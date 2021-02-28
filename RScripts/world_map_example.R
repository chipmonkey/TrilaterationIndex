library('maps')
library("mapproj")
library(tidyverse)
library(lwgeom)
library(spData)
#  Thanks https://ggplot2-book.org/maps.html

samples = read.csv('data/small_sample.csv')
row.names( samples ) <- samples[,1]
samples <- samples[,-1]
summary(samples)
print(samples)

# mymap <- get_map(location = c(lon = mean(samples$y), lat = mean(samples$x)), zoom = 4,
#         maptype = "satellite", scale = 2)

map_world <- map_data('world') %>% 
  select(lon = long, lat, group, id = region)

ggplot(map_world, aes(lon, lat, group=group)) +
  geom_polygon(fill="white", color="grey50") +
  coord_quickmap()
map_world
names(map_world)

# Projections thanks to https://geocompr.robinlovelace.net/reproj-geo-data.html#reprojecting-raster-geometries
world_wintri = lwgeom::st_transform_proj(world, crs = "+proj=wintri")
