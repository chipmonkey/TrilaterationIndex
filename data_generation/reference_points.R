refpoints = c(c('North Pole', 90, 0),
              c('Louisville, KY', 38.26, -85.76),
              c('Sandy Island', -19.22, 159.93),
              c('Aldabra', -9.42, 46.33),
              c('Point Nemo', -48.87, -123.39))

refpoints = c('North Pole', 'Louisville, KY', 'Sandy Island', 'Aldabra', 'Point Nemo',
              90, 38.26, -19.22, -9.42, -48.87,
              0, -85.76, 159.93, 46.33, -123.39)


refpoints <- data.frame(array(refpoints, dim = c(5, 3)))

names(refpoints) <- c('Location', 'Latitude', 'Longitude')

write.csv(refpoints, '../data/ref_points.csv')
