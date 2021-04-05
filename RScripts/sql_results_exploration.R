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
    geom_bar(stat='identity', position=position_dodge()) +
    scale_fill_manual(values=gray.colors(4))

table(data$leftcount)
table(data$rightcount)
# View(data)



library('tidyverse')
View(pivot_longer(data, cols = c('leftcat', 'rightcat')))
data$leftcat <- as.factor(data$leftcat)
data$rightcat <- as.factor(data$rightcat)

data$leftcount <- as.factor(data$leftcount)
data$rightcount <- as.factor(data$rightcount)

pivot_longer(data, cols=c('timing_ms'))

plotdata <- aggregate(data$timing_ms,
          by = list(data$leftcount, data$rightcount, data$algorithm, data$dist),
          FUN=median)
names(plotdata) <- c('leftcount', 'rightcount', 'algorithm', 'dist', 'timing_ms')

library(gridExtra)

p1 <- ggplot(data=plotdata[plotdata$algorithm=='st_distance' & plotdata$dist==0.1,], aes(x=leftcount, y=rightcount)) +
  geom_tile(aes(fill=timing_ms)) +
  geom_text(aes(label=trunc(timing_ms)), colour="black") +
            # colour="green") +
  # scale_fill_continuous(limits=c(0,1000))
  # scale_fill_gradient(low="black", high="white", limits=c(0,1000)) +
  xlab("|P|") +
  ylab("|Q|") +
  scale_fill_gradient2(
    low = "red",
    mid = "white",
    high = "blue",
    midpoint = 300, limits=c(0, 1000),
    space = "Lab",
    na.value = "grey50",
    guide = "colourbar",
    aesthetics = "fill"
  )

  # scale_fill_gradient2(
  #   low = "white",
  #   mid = "gray70",
  #   high = "black",
  #   midpoint = 500, limits=c(0, 1000),
  #   space = "Lab",
  #   na.value = "grey50",
  #   guide = "colourbar",
  #   aesthetics = "fill")

p2 <-  ggplot(data=plotdata[plotdata$algorithm=='Trilateration' & plotdata$dist==0.1,], aes(x=leftcount, y=rightcount)) +
  geom_tile(aes(fill=timing_ms)) +
  geom_text(aes(label=trunc(timing_ms)),
            colour="black") +
  xlab("|P|") +
  ylab("|Q|") +
  # scale_fill_gradient(low = "blue", high = "red")
  # scale_fill_continuous(type='viridis', limits=c(0,1000)) +
  # scale_color_gradient(low='green', high='black', limits=c(0, 1000)) +
  # scale_fill_gradient(low="black", high="white", limits=c(0,1000)) +
  
  scale_fill_gradient2(
    low = "red",
    mid = "white",
    high = "blue",
    midpoint = 300, limits=c(0, 1000),
    space = "Lab",
    na.value = "grey50",
    guide = "colourbar",
    aesthetics = "fill")

  # scale_fill_gradient2(
  #   low = "white",
  #   mid = "gray70",
  #   high = "black",
  #   midpoint = 500, limits=c(0, 1000),
  #   space = "Lab",
  #   na.value = "grey50",
  #   guide = "colourbar",
  #   aesthetics = "fill")
  # scale_fill_gradient2(midpoint = 100, low = "blue", mid = "white", high = "red", space = "Lab" )
  # facet_wrap(~notes)

grid.arrange(p1, p2, nrow=1, ncol=2)

