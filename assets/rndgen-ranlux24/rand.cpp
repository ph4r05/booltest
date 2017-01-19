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

    std::ranlux24 gen(seed);
    fprintf(stderr, "Ranlux24. Min: %lld, Max: %lld\n", (long long)gen.min(), (long long)gen.max());

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE+20];
    for(;;){
        for(int i = 0; i<BUFSIZE; i+=3){
            uint_fast32_t cur = gen();
            buff[i+0] = cur & 0xff;
            buff[i+1] = (cur >> 8)  & 0xff;
            buff[i+2] = (cur >> 16) & 0xff;
        }
        write(fileno(stdout), buff, BUFSIZE);
        fflush(stdout);
    }

    return 0;
}

