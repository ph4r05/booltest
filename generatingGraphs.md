# Graphs in R

Docs:

facet_wrap: http://docs.ggplot2.org/0.9.3.1/facet_wrap.html
Sources: http://www.cookbook-r.com/Graphs/Axes_(ggplot2)/#setting-and-hiding-tick-markers


## Conversion to the CSV for R:

```
python csvgen.py randctest_5MB.txt > /tmp/c5.csv
python csvgen.py --start-idx 3 testjava_1MB_7.txt > /tmp/j1.csv
```

## Boxplots:

```
require(ggplot2)
require(reshape2)

df <- read.csv("/tmp/c5.csv", header=T)
ggplot(data = df, aes(x=variable, y=value)) + geom_boxplot(aes(variable)) + facet_wrap( ~ variable, scales="free", ncol=4, shrink=TRUE) + ylab("Z-score") + xlab("distinguisher")
ggsave(file="/tmp/randc_5mb.pdf", width=2, height=2)

table(df$variable)
barplot(height=table(df$variable))

df <- read.csv("/tmp/j7.csv", header=T)
ggplot(data = df, aes(x=variable, y=value)) + geom_boxplot(aes(variable)) + facet_wrap( ~ variable, scales="free", ncol=4, shrink=TRUE) + ylab("Z-score") + xlab("distinguisher")
ggsave(file="/tmp/randjava_1mb.pdf", width=4, height=3)
```

## Boxplots - randc, java

```
labeller = label_parsed
df <- read.csv("/tmp/test-randc-linux-1MB.csv", header=T)
ggplot(data = df, aes(x=variable, y=value)) + geom_boxplot(aes(variable)) + facet_wrap( ~ variable, scales="free", ncol=4, shrink=TRUE, labeller = label_parsed) + ylab("Z-score") + xlab("distinguisher") + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
ggsave(file="/tmp/randc_1mb.pdf", width=4, height=3)
```

## Axis swap:

```
p + coord_flip()
```

## Fixed coordinates:

```
g + coord_fixed(ratio = 0.2)
```

## Histogram / barplot

```
barplot(height=table(df$variable))
```

## Data size z-score analysis

```
df$V1 <- as.character(df$V1)
df$V1 <- factor(df$V1, levels=unique(df$V1))
library(scales)
ggplot(data = df, aes(x=V1, y=V2)) + geom_boxplot(aes(V1))  + ylab("Z-score") + xlab("data size") + scale_y_continuous(trans=log2_trans())
```

With a bit of tweaking of axis:
```
ggplot(data = df, aes(x=V1, y=V2)) + geom_boxplot(aes(V1))  + ylab("z-score") + xlab("data size") + scale_y_continuous(trans=log2_trans(),breaks=c(2,3,4,6,8,12,16,24,32,48,64)) + geom_hline(aes(yintercept=7.68), colour="#990000", linetype="dashed")
```

## AES size

```
require(ggplot2)
require(reshape2)
df <- read.csv("/tmp/aes2.csv", header=F)
df$V1 <- as.character(df$V1)
df$V1 <- factor(df$V1, levels=unique(df$V1))
ggplot(data = df, aes(x=V1, y=V2)) + geom_boxplot(aes(V1))  + ylab("Z-score") + xlab("data size") + coord_flip()

df <- read.csv("/tmp/aess.csv", header=F)
df$V1 <- as.character(df$V1)
df$V1 <- factor(df$V1, levels=unique(df$V1))
ggplot(data = df, aes(x=V1, y=V2)) + geom_boxplot(aes(V1))  + ylab("Z-score") + xlab("data size") + coord_flip()
```
