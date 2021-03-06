---
title: "Practical Machine Learning Course Project"
author: "Saurav"
date: "9 April 2018"
output:
        html_document:
                keep_md: true

---


In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.


```r
traindestfile = "./data/pml-training.csv"
testdestfile = "./data/pml-testing.csv"
if(!file.exists(traindestfile) || !file.exists(testdestfile)){
        download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = traindestfile)
        download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = testdestfile)
}

training <- read.csv(traindestfile)
testing <- read.csv(testdestfile)
```

## Exploratory Data Analysis



```r
library(ggplot2)
g <- qplot(user_name, data = training, colour = classe)
g
```

<img src="fig/unnamed-chunk-2-1.png" style="display: block; margin: auto;" />

Get the accelerometer data

```r
trainset <- training[,c(1,2,3,4,5,6,7,grep("accel", names(training)), 160)]
```

