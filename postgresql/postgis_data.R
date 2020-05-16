t1 <- read.csv('./data/lat_long_synthetic.csv')
t1$Category <- 1:nrow(t1)
head(t1)
t1$Category <- t1$Category %% 10

# Create categories with varied sizes:
# 100, 1000, 2000, 5000, 10000
# The last category of 32000 is really 
t1[1:100, 'Category'] <- 10
t1[101:1000, 'Category'] <- 11
t1[1001:3000, 'Category'] <- 12
t1[3001:8000, 'Category'] <- 13
t1[8001:18000, 'Category'] <- 14
t1[18001:50000, 'Category'] <- 15

head(t1$Category)
summary(t1$Category)
table(t1$Category)

write.csv(t1, file='./data/lat_long_categorized.csv', quote = FALSE, row.names = FALSE)

plot(t1$Latitude, t1$Longitude, col=t1$Category)

library('maps')
library("ggplot2")
usa <- map_data("usa") 
states <- map_data("state")

ggplot() + 
  geom_point(data = t1, aes(x=Longitude, y = Latitude), fill = NA, color = t1$Category) + 
  coord_fixed(1.3)

table(t1$Category)
class(t1$Category)
