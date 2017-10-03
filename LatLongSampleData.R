x <- read.csv(file = 'lat_long_data.csv', encoding='UTF-8')
names(x) <- c('Latitude', 'Longitude')

x2 <- data.frame(Latitude = jitter(x$Latitude, factor = 1))
x2$Longitude<- jitter(x$Longitude, factor = 1)

# x2$Latitude <- x2$Latitude + runif(n = nrow(zz), min = 0.001, max = 0.004)
# x2$Longitude <- x2$Longitude - runif(n = nrow(zz), min = 0.001, max = 0.004)


# Guarantee that points are NOT in their original place:
l <- which((x$Latitude-x2$Latitude) < 0.001 & (x$Latitude-x2$Latitude) > 0)
x2[l,'Latitude'] <- x2[l, 'Latitude'] + runif(n = length(l), min = 0.001, max = 0.004)

l <- which(abs(x$Latitude-x2$Latitude) < 0.001 & (x$Latitude-x2$Latitude) <= 0)
x2[l,'Latitude'] <- x2[l, 'Latitude'] - runif(n = length(l), min = 0.001, max = 0.004)

l <- which((x$Longitude-x2$Longitude) < 0.001 & (x$Longitude-x2$Longitude) > 0)
x2[l,'Longitude'] <- x2[l, 'Longitude'] + runif(n = length(l), min = 0.001, max = 0.004)

l <- which(abs(x$Longitude-x2$Longitude) < 0.001 & (x$Longitude-x2$Longitude) <= 0)
x2[l,'Longitude'] <- x2[l, 'Longitude'] - runif(n = length(l), min = 0.001, max = 0.004)

plot(x$Latitude-x2$Latitude, x$Longitude-x2$Longitude)

# Select a round number of records:
nrow(x2)
x2 <- x2[1:150000,]

x2$Latitude <- round(x2$Latitude, digits = 7)
x2$Longitude <- round(x2$Longitude, digits = 7)
plot(density(x$Latitude))
plot(density(x$Longitude))

write.csv(x2, file = 'lat_long_synthetic.csv', row.names = FALSE)

plot(x2$Latitude, x2$Longitude)
