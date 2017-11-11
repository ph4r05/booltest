/**
 * Generating random numbers in C using simple rand() function
 * gcc -o rand -Wall rand.c
 */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

/**
 * Returns an integer in the range [0, n).
 *
 * Uses rand(), and so is affected-by/affects the same seed.
 */
int randint(int n) {
  if ((n - 1) == RAND_MAX) {
    return rand();
  } else {
    // Chop off all of the values that would cause skew...
    long end = RAND_MAX / n; // truncate skew
    assert (end > 0L);
    end *= n;

    // ... and ignore results from rand() that fall above that limit.
    // (Worst case the loop condition should succeed 50% of the time,
    // so we can expect to bail out of this loop pretty quickly.)
    int r;
    while ((r = rand()) >= end);

    return r % n;
  }
}

#define BITW 31
#define BUFSIZE 1024

int main(int argc, char * argv[]){
    if (argc == 1){
        srand(0);
    } else {
        srand(atoi(argv[1]));
    }

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE+16];
    for(;;){
        unsigned int pos = 0;
        while(pos < BUFSIZE * 8){
            unsigned int x = (unsigned int) rand();
            int avail = BITW;


        }

        for(int i = 0; i<BUFSIZE; i+=3){
            unsigned int x = (unsigned int) rand();


            buff[i] = x & 0xff;
            buff[i+1] = (x >> 8) & 0xff;
            buff[i+2] = (x >> 16) & 0xff;
        }
        write(fileno(stdout), buff, BUFSIZE);
	    fflush(stdout);
    }

    return 0;
}
