library(ggplot2)
library(data.table)

setwd("~/repos/TrilaterationIndex")

data <- read.csv('./postgresql/query_timings.csv')
table(data$notes)
data$dist <- 0
data[data$notes %like% 'dist: 0.1','dist'] <- 0.1
data[data$notes %like% 'dist: 1','dist'] <- 1
data[data$notes %like% 'dist: 10','dist'] <- 10
data[data$notes %like% 'dist: 100','dist'] <- 100

data <- data[data$notes %like% 'Adequ',]
data$algorithm = 'None'
data[data$notes %like% 'CatAdequacy','algorithm'] = 'st_distance'
data[data$notes %like% 'CatTrilatAdequacy', 'algorithm'] = 'Trilateration'
table(data$algorithm)
data <- data[data$algorithm != 'None',]
data$dist <- as.factor(data$dist)

ggplot(data=data, aes(x=dist, y=timing_ms, fill=algorithm)) +
    geom_bar(stat='identity', position=position_dodge())
  