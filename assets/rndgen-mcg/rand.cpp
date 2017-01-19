/**
 * Generating random numbers in C using Mersenne Twister 19937
 * gcc -o rand -Wall rand.c
 */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <random>
#define BUFSIZE 1024

int main(int argc, char * argv[]){
    int seed = 0;
    if (argc > 1){
        seed = atoi(argv[1]);
    }

    std::minstd_rand gen(seed);
    fprintf(stderr, "minstd. Min: %lld, Max: %lld\n", (long long)gen.min(), (long long)gen.max());

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE];
    for(;;){
        for(int i = 0; i<BUFSIZE; i+=2){
            uint_fast32_t cur = gen() - 1;
            buff[i+0] = cur & 0xff;
            buff[i+1] = (cur >> 8)  & 0xff;
        }
        write(fileno(stdout), buff, BUFSIZE);
        fflush(stdout);
    }

    return 0;
}

