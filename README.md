# Polyverif

## Local installation

From the local dir:

```
pip install --upgrade --find-links=. .
```


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

## Python 2.7.13

It won't work with lower Python version. Use `pyenv` to install a new Python version.
It internally downloads Python sources and installs it to `~/.pyenv`.

```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec $SHELL
pyenv install 2.7.13
pyenv local 2.7.13
```

## GCC 5.2

Installing a new GCC with C++ 11 support.
http://bakeronit.com/2015/11/04/install_gcc/

```
wget http://ftp.gnu.org/gnu/gcc/gcc-5.2.0/gcc-5.2.0.tar.bz2
tar -xjvf gcc-5.2.0.tar.bz2

module add mpc-0.8.2
module add gmp-4.3.2
module add mpfr-3.0.0

mkdir -p ~/local/gcc-5.2.0
cd local
mkdir gcc-build  # objdir
cd gcc-build
../../gcc-5.2.0/configure --prefix=~/local/gcc-5.2.0/ --enable-languages=c,c++,fortran,go --disable-multilib
make -j4 # spend a long time
make install

# Add either to ~/.bashrc or just invoke on shell
export PATH=~/local/gcc-5.2.0/bin:$PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib64:$LD_LIBRARY_PATH
```

## Compiling EACirc generator on Aura/Aisa

```
module add mpc-0.8.2
module add gmp-4.3.2
module add mpfr-3.0.0
module add cmake-3.6.2
export PATH=~/local/gcc-5.2.0/bin:$PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib64:$LD_LIBRARY_PATH

cd ~/eacirc
mkdir -p build && cd build
CC=gcc CXX=g++ cmake ..
make
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

```
openjdk version "1.8.0_121"
OpenJDK Runtime Environment (build 1.8.0_121-8u121-b13-0ubuntu1.16.04.2-b13)
OpenJDK 64-Bit Server VM (build 25.121-b13, mixed mode)
Ubuntu 16.04.1 LTS (Xenial Xerus)
```


## Egenerator speed benchmark

Table summarizes function & time needed to generate 10 MB of data.

| Function      | Round | Time (sec)
| ------------- | ----- | --------------|
| AES           |  4    | 2.12984800339 |
| ARIRANG       |  4    | 9.43074584007 |
| AURORA        |  5    | 0.810596942902 |
| BLAKE         |  3    | 0.839290142059 |
| Cheetah       |  7    | 0.924134969711 |
| CubeHash      |  3    | 36.8423719406 |
| DCH           |  3    | 3.34326887131 |
| DECIM         |  7    | 51.946573019 |
| DynamicSHA    |  9    | 1.33032679558 |
| DynamicSHA2   |  14   | 1.14816212654 |
| ECHO          |  4    | 2.15773296356 |
| Fubuki        |  4    | 1.81450080872 |
| Grain         |  4    | 67.9190270901 |
| Grostl        |  5    | 2.10276603699 |
| Hamsi         |  3    | 7.09616398811 |
| Hermes        |  3    | 1.46782112122 |
| JH            |  8    | 3.51690793037 |
| Keccak        |  4    | 1.31340193748 |
| Lesamnta      |  5    | 2.08995699883 |
| LEX           |  5    | 0.789785861969 |
| Luffa         |  8    | 2.70372700691 |
| MD6           |  11   | 2.13406395912 |
| Salsa20       |  4    | 0.845487833023 |
| SIMD          |  3    | 7.54037189484 |
| Tangle        |  25   | 1.43553209305 |
| TEA           |  8    | 0.981395959854 |
| TSC-4         |  14   | 8.33323192596 |
| Twister       |  9    | 1.38356399536 |

