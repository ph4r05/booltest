/**
 * Generating random numbers in C using Mersenne Twister with 2^{127} - 1 period
 * length using TinyMT project (https://github.com/MersenneTwister-Lab/TinyMT)
 * gcc -o rand -Wall rand.c
 */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include "tinymt32.h"

#define BUFSIZE 1024

int main(int argc, char * argv[]){
    tinymt32_t state;
    if (argc == 1){
	tinymt32_init(&state, time(NULL));
    } else {
        tinymt32_init(&state, atoi(argv[1]));
    }

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE];
    for(;;){
        for(int i = 0; i<BUFSIZE; i+=4){
            uint32_t cur = tinymt32_generate_uint32(&state);
	    buff[i+0] = cur & 0xff;
	    buff[i+1] = (cur >> 8)  & 0xff;
	    buff[i+2] = (cur >> 16) & 0xff;
	    buff[i+3] = (cur >> 24) & 0xff;
        }
        write(fileno(stdout), buff, BUFSIZE);
	fflush(stdout);
    }

    return 0;
}

