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

//    for(int i=0; i<4;i++){
//    printf("  %x\n", state.status[i]);
//    }

    if (argc == 1){
	    tinymt32_init(&state, 0);
    } else {
        tinymt32_init(&state, (uint32_t)atoi(argv[1]));
    }

//    printf("%x\n", TINYMT32_MASK);
//    for(int i=0; i<4;i++){
//    printf("  %x\n", state.status[i]);
//    }

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE+64];
    for(;;){
        for(int i = 0; i<BUFSIZE; i+=4){
            uint32_t cur = tinymt32_generate_uint32(&state);
//            printf("%x\n", cur);
//            return 1;
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

