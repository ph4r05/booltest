# Polyverif

## Scipy installation with pip

```
pip install pyopenssl
pip install pycrypto
pip install git+https://github.com/scipy/scipy.git
pip install --upgrade --find-links=. .
```

# Graphs in R

Docs:

facet_wrap: http://docs.ggplot2.org/0.9.3.1/facet_wrap.html
Sources: http://www.cookbook-r.com/Graphs/Axes_(ggplot2)/#setting-and-hiding-tick-markers


## Boxplots:

```
df <- read.csv("/tmp/c10.csv", header=T)
ggplot(data = df, aes(x=variable, y=value)) + geom_boxplot(aes(variable)) + facet_wrap( ~ variable, scales="free", ncol=4, shrink=TRUE) + ylab("z-score") + xlab("distinguisher")
```

## Axis swap:

```
p + coord_flip()
```

## Histogram / barplot

```
barplot(height=table(df$variable))
```

## Data size analysis

```
df$V1 <- as.character(df$V1)
df$V1 <- factor(df$V1, levels=unique(df$V1))
library(scales)
ggplot(data = df, aes(x=V1, y=V2)) + geom_boxplot(aes(V1))  + ylab("z-score") + xlab("data size") + scale_y_continuous(trans=log2_trans())
```

# Java tests - version

openjdk version "1.8.0_121"
OpenJDK Runtime Environment (build 1.8.0_121-8u121-b13-0ubuntu1.16.04.2-b13)
OpenJDK 64-Bit Server VM (build 25.121-b13, mixed mode)
Ubuntu 16.04.1 LTS (Xenial Xerus)


