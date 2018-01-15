library('RPostgreSQL')

drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, dbname = "trilateration", host = "localhost", port = 5433)
?dbDriver
?`dbDriver-methods`

qt <- read.csv('./data/query_timings.csv')
head(qt)
summary(qt)

cats <- read.csv('./data/category_counts.csv')
head(cats)

t2 <- merge(x = qt, y=cats, by.x = 'leftcat', by.y='category')
t3 <- merge(x=t2, y=cats, by.x = 'rightcat', by.y = 'category')
names(t3)

library(rgl)
example(plot3d)
