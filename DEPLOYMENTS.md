## Aura / Aisa on FI MU

```
module add cmake-3.6.2
module add gcc-4.8.2
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

