/**
 * Generating random numbers in C using simple rand() function
 * gcc -o rand -Wall rand.c
 */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>


#define BUFSIZE 1024

int main(int argc, char * argv[]){
    if (argc == 1){
        srand(0);
    } else {
        srand(atoi(argv[1]));
    }

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE];
    for(;;){
        for(int i = 0; i<BUFSIZE; i+=2){
            int x = rand();
            buff[i] = (x >> 8) & 0xff;
            buff[i+1] = x & 0xff;
        }
        write(fileno(stdout), buff, BUFSIZE);
	    fflush(stdout);
    }

    return 0;
}
