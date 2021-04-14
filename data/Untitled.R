library(tidyverse)
library(caret)

df <- read.csv("train.csv", nrows=1000)

train.index <- createDataPartition(df$label, p = .7, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

# pixel range from 0-255
x_train <- train %>% select(- label) 
y_train <- train %>% select(label)

x_test <- test %>% select(- label)
y_test <- test %>% select(label)

x_train <- as.matrix(x_train) / 255
x_test <- as.matrix(x_test) / 255

pca <- prcomp(x_train, rank=128)
x_train_pca <- x_train %*% pca$rotation
x_test_pca <- x_test %*% pca$rotation