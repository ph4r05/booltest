# BoolTest

[![Build Status](https://travis-ci.org/ph4r05/polynomial-distinguishers.svg?branch=master)](https://travis-ci.org/ph4r05/polynomial-distinguishers)

Boolean PRNG tester - analysing statistical properties of PRNGs.

Randomness tester based on our paper published at [Secrypt 2017](https://crocs.fi.muni.cz/public/papers/secrypt2017)

## How does it work?

BoolTest generates a set of boolean functions, computes the expected result distribution when evaluated on truly random
data and compares this to the evaluation on the data being tested. 

## Pip installation

BoolTest is available via `pip`:

```
pip3 install booltest
```

## Local installation

From the local dir:

```
pip3 install --upgrade --find-links=. .
```

## The engine

BoolTest does the heavy lifting with the native python extension [bitarray_ph4](https://github.com/ph4r05/bitarray)

Bitarray operations are performed effectively using fast operations implemented in C.


# Experiments

## First launch

The following commands generate two different files, random and zero-filled.
Both are tested, the difference between files should be evident.

```
dd if=/dev/urandom of=random-file.bin bs=1024 count=$((1024*10))
dd if=/dev/zero of=zero-file.bin bs=1024 count=$((1024*10))

booltest --degree 2 --block 256 --combine-deg 2 --top 128 --tv $((1024*1024*10)) --rounds 0 random-file.bin
booltest --degree 2 --block 256 --combine-deg 2 --top 128 --tv $((1024*1024*10)) --rounds 0 zero-file.bin
```

- The BoolTest with the given parameters constructs all polynomials of degree 2 
from monomials {x_0, ..., x_{255}}
- Evaluates all polynomials on the input data (windowing), computes zscore from the 
computed vs reference data 
- Selects 128 best polynomials (abs(zscore))
- Phase 2: Take the best 128 polynomials and combine them by XOR 
to the `--combine-deg` number of terms. 
- The resulting polynomials are evaluated again and results printed out.

## Common testing parameters

We usually use BoolTest with the following testing parameters:

```
--top 128 --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256
```

The same can be done with the `--default-params`

## Output and p-values

BoolTest returns zscores of the best distinguishers.

In order to obtain a p-value from the Z-score you need to compute a reference experiments, i.e., compute N BoolTest experiments on a random data and observe the z-score distribution.
Z-score is data-size invariant but it depends on the BoolTest parameters `(n, deg, k)`.

The most straightforward evaluation is to check whether z-score obtained from the real experiment has been observed in the reference runs. 
If not, we can conclude the BoolTest rejects the null hypothesis with pvalue `1/N`.

To obtain lower alpha you need to perform more reference experiments, 
to obtain higher alpha integrate the z-score histogram from tails to mean to obtain desired percentage of the area under z-score histogram.

The file [pval_db.json](https://github.com/ph4r05/polynomial-distinguishers/blob/master/pval_db.json) contains reference z-score -> pvalue mapping for N=20 000 reference runs.

BoolTest now supports adding pvalue database as a parameter `--ref-db path-to-db.json`
If the database is not given, BoolTest tries to locate the default `pval_db.json` in the BoolTest installation directory and on the path.

If the database is found, BoolTest shows also OK/reject result for the best distinguisher, given the reference database contains the 
data for given `(n, deg, k)` parameters.

Example:
```
 - best poly zscore  -5.37867, expp: 0.0625, exp:   10240, obs:    9713, diff:  5.1464844 %, poly: [[64, 245, 207, 242]]
2019-12-13 20:25:17 PHX booltest.booltest_main[51363] INFO Ref samples: 40005, min-zscrore: 4.838657, max-zscore: 7.835336, best observed: 5.3786712268614005, rejected: False, alpha: 2.4996875390576178e-05
```

## Halving method

We have implemented another evaluation method called halving, enabled with commandline option `--halving`.
It needs twice more data than the default method, because of how it works:

- The input file is divided to two halves
- BoolTest runs as before on the first half, picks the best distinguisher
- BoolTest runs the best distinguisher on the second half
- As the best distinguisher selected to the second half "never seen" 
the second half and there is only one polynomial the p-value can be directly computed due to independence.

The best distinguisher results are essentially following Binomial distribution:
`Bi(number_of_blocks, probability_of_dist_eval_to_1)`.

To compute the p-value we run the Binomial test:
```scipy.stats.binom_test(observed_ones, n=ntrials, p=dist_probab, alternative='two-sided')``` 
 
This method eliminates a need to have a `pval_db.json` database computed with the reference data for given parameters.
The benefit is the halving method gives directly a p-value, without a need to run reference computations.
The downside is the method needs twice more data and can give weaker results than the original BoolTest evaluation.

Example:
```
 - zscore[idx00]: -0.40825, observed: 00010200, expected: 00010240   idx:      0, poly: [[64, 245, 207, 242]]
2019-12-13 20:25:17 PHX booltest.booltest_main[51363] INFO Binomial dist, two-sided pval: 0.6868421673496484, pst: 0.0625, ntrials: 163840, succ: 10200
```

## Java random

Analyze output of the `java.util.Random`, use only polynomials in the specified file. Analyze 100 MB of data:

```
booltest --degree 2 --block 512 --combine-deg 2 --top 128 --tv $((1024*1024*100)) --rounds 0 \
  --poly-file data/polynomials/polynomials-randjava_seed0.txt \
  randjava_seed0.bin
```

## Input data

BoolTest can test:

- Pregenerated data files
- Use the [CryptoStreams] configuration files to generate input data on the fly, using [CryptoStreams] (library contains plenty round-reduced cryptographic primitives)
 
## Cluster computation (Metacentrum)

- Map / Reduce. 
  - The `booltest/testjobs.py` creates job files
  - The `booltest/testjobsproc.py` processes result files
- BoolTest job is configured via JSON file. Result of a computation is JSON file.
- The `booltest/testjobsbase.py` performs job aggregation, i.e., more BoolTest runs in one shell script as job planning overhead is non-negligible. Useful for fast running jobs.
- Works with PBSPro, qsub queueing algorithm


### Example - generate jobs from [CryptoStreams] configurations

```bash
python ../booltest/booltest/testjobs.py  \
    --data-dir $RESDIR --job-dir $JOBDIR --result-dir=$RESDIR \
    --top 128 --matrix-size 1 10 100 --matrix-block 128 256 384 512 --matrix-deg 1 2 3 --matrix-comb-deg 1 2 3 \
    --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256 \
    --skip-finished --no-functions --ignore-existing \
    --generator-folder ../bool-cfggens/ --generator-path ../bool-cfggens/crypto-streams_v2.3-13-gff877be
```

For all [CryptoStreams] configuration files located under `../bool-cfggens/` it generates BoolTest tests
with parameters: 

```
input_size x block_size x deg x comb-deg
{1, 10, 100} x {128, 256, 384, 512} x {1, 2, 3} x {1, 2, 3}
```

- Command generates PBSPro shell scripts to `$JOBDIR`, results are placed into `$RESDIR`.
- For one configuration file which is typically round reduced crypto primitive it performs `3*4*3*3 = 108 tests`.
- When using CryptoStreams config files the config files have to specify the longest tested input, in this case, 100 MB.


### Example - analyze input files

```bash
python ../booltest/booltest/testjobs.py  \
    --test-files ../card_prng/*.bin \
    --data-dir $RESDIR --job-dir $JOBDIR --result-dir=$RESDIR \
    --top 128 --matrix-size 1 10 100 --matrix-block 128 256 384 512 --matrix-deg 1 2 3 --matrix-comb-deg 1 2 3 \
    --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256 \
    --skip-finished --no-functions --ignore-existing 
```

This example generates job to analyze input files (e.g., smartcard generated randomness)


### Example - reference statistics

```bash
python ../booltest/booltest/testjobs.py  \
    --data-dir $RESDIR --job-dir $JOBDIR --result-dir=$RESDIR \
    --generator-path --generator-path ../bool-cfggens/crypto-streams_v2.3-13-gff877be \
    --top 128 --matrix-size 10 --matrix-block 128 256 384 512 --matrix-deg 1 2 3 --matrix-comb-deg 1 2 3 \
    --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256 \
    --skip-finished --ref-only --test-rand-runs 1000 --skip-existing --counters-only --no-sac --no-rpcs --no-reinit
```

Computes 1000 independent AES round 10 runs, each with different seed in the counter mode. 
Tests BoolTest in various configurations.

## Reference statistics (old)

In order to test reference statistics of the test we computed polynomial tests on input vectors generated by
`AES-CTR(SHA256(random_32bit()))` - considered as random data source. The `randverif.py` was used.

The first hypothesis to verify is the following: under null hypothesis (uniform input data), zscore test is input
data size invariant. In other words, the zscore result of the test is not influenced by amount of data processed.

To verify the first hypothesis we analyzed 1000 different test vectors of sizes 1 and 10 MB for various settings
(`block \in {128, 256} x deg \in {1, 2, 3} x comb_deg \in {1, 2, 3}`) and compared results. The test was performed with
`assets/test-aes-size.sh`.

Second test is to determine reference zscore value for random data. For this we performed 100 different tests on 10 MB
AES input vectors in all test combinations: `block \in {128, 256, 384, 512} x deg \in {1, 2, 3} x comb_deg \in {1, 2, 3}`.


## Aura testbed (old)

Testbed = battery of functions (e.g., ESTREAM, SHA3 candidates, ...) tested with various polynomial parameters
(e.g., `block \in {128, 256, 384, 512} x deg \in {1, 2, 3} x comb_deg \in {1, 2, 3}`).

EAcirc generator is invoked during the test to generate output from battery functions. If switch `--data-dir` is used
`testbed.py` will try to look up output there first.

In order to start EACirc generator you may need to compile it on the machine you want to test on. Instructions
 for compilation are on the bottom of the page. In order to invoke the generator you need to setup env

```
module add mpc-0.8.2
module add gmp-4.3.2
module add mpfr-3.0.0
module add cmake-3.6.2
export PATH=~/local/gcc-5.2.0/bin:$PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib64:$LD_LIBRARY_PATH
```

In order to start `testbed.py` there is a script `assets/aura-para.sh`. It performs the env setup, prepares directories,
spawns multiple testing processes.

Parallelization is done in a simple way. Each test has an index. This order is randomized and each process from the
batch takes the job that belongs to him (e.g. 10 processes, process #5 takes each 5th job). If the ordering is not
favorable for in some way (e.g., one process is getting too much heavy jobs - deg3, combdeg 3) just change the seed
of the test randomizer.

Result of each test is stored in a separate file.

## Standard functions -> batteries

The goal of this experiment is to assess standard test batteries (e.g., NIST, Dieharder, TestU01) how well they perform
on the battery of round reduced functions (e.g., ESTREAM, SHA3 candidates, ...)

For the testing we use Randomness Testing Toolkit (RTT) from the EACirc project. The `testbatteries.py` prepares data
for functions to test and the main bash script that submits tests to RTT.

```
python booltest/testbatteries.py --email ph4r05@gmail.com --threads 3 \
    --generator-path ~/eacirc/generator/generator \
    --result-dir ~/_nni/home/ph4r05/testdata/ \
    --data-dir ~/_nni/home/ph4r05/testdata/ \
    --script-data /home/ph4r05/testdata \
    --matrix-size 1 10 100 1000
```

## RandC

Test found distinguishers on RandC for 1000 different random seeds:

```
python booltest/randverif.py --test-randc \
    --block 384 --deg 2 \
    --tv $((1024*1024*10)) --rounds 0 --tests 1000 \
    --poly-file polynomials-randc-linux.txt \
    > ~/output.txt
```

In order to generate CSV from the output:

```
python csvgen.py output.txt > data.csv
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
| ------------- | ----- | ---------------|
| AES           |  4    | 2.12984800339  |
| ARIRANG       |  4    | 9.43074584007  |
| AURORA        |  5    | 0.810596942902 |
| BLAKE         |  3    | 0.839290142059 |
| Cheetah       |  7    | 0.924134969711 |
| CubeHash      |  3    | 36.8423719406  |
| DCH           |  3    | 3.34326887131  |
| DECIM         |  7    | 51.946573019   |
| DynamicSHA    |  9    | 1.33032679558  |
| DynamicSHA2   |  14   | 1.14816212654  |
| ECHO          |  4    | 2.15773296356  |
| Fubuki        |  4    | 1.81450080872  |
| Grain         |  4    | 67.9190270901  |
| Grostl        |  5    | 2.10276603699  |
| Hamsi         |  3    | 7.09616398811  |
| Hermes        |  3    | 1.46782112122  |
| JH            |  8    | 3.51690793037  |
| Keccak        |  4    | 1.31340193748  |
| Lesamnta      |  5    | 2.08995699883  |
| LEX           |  5    | 0.789785861969 |
| Luffa         |  8    | 2.70372700691  |
| MD6           |  11   | 2.13406395912  |
| Salsa20       |  4    | 0.845487833023 |
| SIMD          |  3    | 7.54037189484  |
| Tangle        |  25   | 1.43553209305  |
| TEA           |  8    | 0.981395959854 |
| TSC-4         |  14   | 8.33323192596  |
| Twister       |  9    | 1.38356399536  |


# Installation

## Scipy installation with pip

```
pip install pyopenssl
pip install pycrypto
pip install git+https://github.com/scipy/scipy.git
pip install --upgrade --find-links=. .
```

## Virtual environment

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

## Python 2.7.14+

BoolTest does not work with lower Python version. Use `pyenv` to install a new Python version.
It internally downloads Python sources and installs it to `~/.pyenv`.

```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec $SHELL
pyenv install 2.7.14
pyenv local 2.7.14
```

The recommended version is Python 3.5+

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


[CryptoStreams]: https://github.com/crocs-muni/CryptoStreams
