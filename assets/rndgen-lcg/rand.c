/**
 * Simple linear congruential generator
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>

/* always assuming int is at least 32 bits */
int lcgrand();
int rseed = 0;
#define BUFSIZE 1024

inline void lcgsrand(int x)
{
    rseed = x;
}
 
#define LCG_RAND_MAX_32 ((1U << 31) - 1)
#define LCG_RAND_MAX ((1U << 15) - 1)
 
inline int lcgrand()
{
    return (rseed = (rseed * 214013 + 2531011) & LCG_RAND_MAX_32);
}
 
int main(int argc, char * argv[])
{
    if (argc > 1){
        lcgsrand(atoi(argv[1]));
    }
    
    // Generates random numbers ad infinitum.
    char buff[BUFSIZE+24];
    for(;;){
        for(int i = 0; i<BUFSIZE; i+=3){
            uint32_t cur = (uint32_t) lcgrand();
            buff[i+0] = cur & 0xff;
            buff[i+1] = (cur >> 8)  & 0xff;
            buff[i+2] = (cur >> 16) & 0xff;
            //buff[i+3] = (cur >> 24) & 0xff;
        }
        write(fileno(stdout), buff, BUFSIZE);
	    fflush(stdout);
    }

    return 0;
}

