/**
 * Generating random numbers in C using Mersenne Twisters - various twisters, .
 * http://www.iro.umontreal.ca/~simardr/testu01/guideshorttestu01.pdf
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/ARTICLES/tgfsr3.pdf
 */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "u01/usoft.h"
#include "u01/ugfsr.h"

#define BUFSIZE 1024

/* S[] initialization with seed */
static void init_genrand (unsigned long * S, unsigned cnt, unsigned long s)
{
   S[0] = s & 0xffffffffUL;
   for (unsigned j = 1; j < cnt; j++) {
      S[j] = (1812433253UL * (S[j - 1] ^ (S[j - 1] >> 30)) + j);
      /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
      /* In the previous versions, MSBs of the seed affect   */
      /* only MSBs of the array state->X[].                  */
      /* 2002/01/09 modified by Makoto Matsumoto             */
      S[j] &= 0xffffffffUL;
      /* for >32 bit machines */
   }
}

int main(int argc, char * argv[]){
    unsigned long S[10000];
    unsigned long seed = 1;
    if (argc > 1){
        seed = atoi(argv[1]);
    }

    // Seed generators with S[]
    init_genrand(S, 10000, seed);

    // Generators
#if defined(TT400)
    unif01_Gen * gen = ugfsr_CreateTT400(S);
#   define GENWIDTH 2
#   define GENOFFSET 0

#elif defined(TT403)
    unif01_Gen * gen = ugfsr_CreateTT403(S);
#   define GENWIDTH 3
#   define GENOFFSET 0

#elif defined(TT775)
    unif01_Gen * gen = ugfsr_CreateTT775(S);
#   define GENWIDTH 3
#   define GENOFFSET 0

#elif defined(TT800)
    unif01_Gen * gen = ugfsr_CreateTT800(S);
#   define GENWIDTH 4
#   define GENOFFSET 0

#elif defined(T800)
    unif01_Gen * gen = ugfsr_CreateT800(S);
#   define GENWIDTH 4
#   define GENOFFSET 0

#elif defined(TOOT73)
    unif01_Gen * gen = ugfsr_CreateToot73(S);
#   define GENWIDTH 2
#   define GENOFFSET 8  // reaaly weird, bottom 8 bits are 0

#elif defined(KIRK81)
    unif01_Gen * gen = ugfsr_CreateKirk81(seed);
#   define GENWIDTH 4
#   define GENOFFSET 0

#elif defined(RIPLEY90)
    unif01_Gen * gen = ugfsr_CreateRipley90(seed);
#   define GENWIDTH 3
#   define GENOFFSET 0

#elif defined(FUSHIMI90)
    unif01_Gen * gen = ugfsr_CreateFushimi90((int)seed);
#   define GENWIDTH 3
#   define GENOFFSET 0

#elif defined(ZIFF98)
    unif01_Gen * gen = ugfsr_CreateZiff98(S);
#   define GENWIDTH 4
#   define GENOFFSET 0

#else
    unif01_Gen * gen = NULL;
#   define GENWIDTH 2
#   define GENOFFSET 0
#   error No generator defined
#endif

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE+128];
    for(;;){
        for(int i = 0; i<BUFSIZE; i+=GENWIDTH){
            unsigned long cur = gen->GetBits(gen->param, gen->state);
            buff[i+0] = (cur >> GENOFFSET) & 0xff;
#if GENWIDTH >= 2
            buff[i+1] = (cur >> (GENOFFSET+8))  & 0xff;
#if GENWIDTH >= 3
            buff[i+2] = (cur >> (GENOFFSET+16)) & 0xff;
#if GENWIDTH == 4
            buff[i+3] = (cur >> (GENOFFSET+24)) & 0xff;
#endif
#endif
#endif
        }
        write(fileno(stdout), buff, BUFSIZE);
        fflush(stdout);
    }

    ugfsr_DeleteGen(gen);
    return 0;
}

