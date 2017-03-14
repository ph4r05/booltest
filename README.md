# Polyverif

## Scipy installation with pip

```
pip install pyopenssl
pip install pycrypto
pip install git+https://github.com/scipy/scipy.git
pip install --upgrade --find-links=. .
```

# Virtual environment

It is usually recommended to create a new python virtual environment for the project:

```
virtualenv ~/pyenv
source ~/pyenv/bin/activate
pip install --upgrade pip
pip install --upgrade --find-links=. .
```

## Aura / Aisa on FI MU

```
module add cmake-3.6.2
module add gcc-4.8.2
```

## Python 2.7 at least

It wont work with lower Python version

```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec $SHELL
pyenv install 2.7.13
```


# Graphs in R

Docs:

facet_wrap: http://docs.ggplot2.org/0.9.3.1/facet_wrap.html
Sources: http://www.cookbook-r.com/Graphs/Axes_(ggplot2)/#setting-and-hiding-tick-markers


## Conversion to the CSV for R:

```
python csvgen.py randctest_5MB.txt > /tmp/c5.csv
python csvgen.py --start-char c testjava_1MB_7.txt > /tmp/j1.csv
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

# Java tests - version

openjdk version "1.8.0_121"
OpenJDK Runtime Environment (build 1.8.0_121-8u121-b13-0ubuntu1.16.04.2-b13)
OpenJDK 64-Bit Server VM (build 25.121-b13, mixed mode)
Ubuntu 16.04.1 LTS (Xenial Xerus)


