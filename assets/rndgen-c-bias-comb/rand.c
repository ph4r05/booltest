/**
 * Generating random numbers in C using simple rand() function
 * gcc -o rand -Wall rand.c
 */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <assert.h>

#define BIAS_OP_XOR 1
#define BIAS_OP_AND 2

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

#define BUFSIZE 1024

#define ZERO_CHANCE 1  // 1:10
#define BLOCKLEN 16    // 16B
#define BIAS_OP BIAS_OP_XOR
#define DEG 2          // OP degree

// Byte index of bit
#define BITBYTE(X) (X/8)
#define BITSHIFT(X) (7-((X) % 8))
#define BITMASK(X) (1 << BITSHIFT(X))
#define BITVAL(buff, off, bit) (((((int)(buff)[off + BITBYTE(bit)]) & BITMASK(bit)) >> BITSHIFT(bit)) & 0x1 )


int main(int argc, char * argv[]){
    int zero_chance = ZERO_CHANCE;
    int blocklen = BLOCKLEN;
    int bias_op = BIAS_OP;
    int deg = DEG;
    int bit_pos[] = {0, 42, 100, 127, 91};
    const int num_vars = sizeof(bit_pos)/sizeof(bit_pos[0]);

    // usage: seed operation prob deg
    if (argc != 5){
        printf("usage: %s seed operation prob degree\n", argv[0]);
        printf("  - seed        = number\n");
        printf("  - operation   = %d for XOR, %d for AND\n", BIAS_OP_XOR, BIAS_OP_AND);
        printf("  - prob        = 1/x chance to bias term T\n");
        printf("  - degree      = degree of term T to keep zero\n\n");
        printf("Each predefined term T on a block len %d bytes is kept with 1/prob\n", blocklen);
        printf("a zero value. Term variables are fixed by now. \n\n");
        printf("Fixed term variables:");
        for(int ctr=0; ctr<num_vars; ctr++){
            printf(" %03d", bit_pos[ctr]);
        }
        printf("\n\n");
        return 1;
    } else {
        srand(atoi(argv[1]));
        bias_op = atoi(argv[2]);
        zero_chance = atoi(argv[3]);
        deg = atoi(argv[4]);

        char * op_sign = bias_op == BIAS_OP_XOR ? "^" : "&";
        fprintf(stderr, "Generating with seed=%d, op=%s, prob=1/%d, deg=%d, block=%d B\n", atoi(argv[1]),
                bias_op == BIAS_OP_XOR ? "XOR":"AND", zero_chance, deg, blocklen);
        fprintf(stderr, "Keeping term T =");
        for(int ctr=0; ctr<deg; ctr++){
            fprintf(stderr, " x_%03d %s", bit_pos[ctr], ctr+1 < deg ? op_sign : "");
        }
        fprintf(stderr, "= 0 with probability 1/%d\n", zero_chance);
    }

    // Checks
    assert(deg <= sizeof(bit_pos)/sizeof(bit_pos[0]));
    assert(BIAS_OP_XOR != BIAS_OP_AND);
    assert(bias_op == BIAS_OP_XOR || bias_op == BIAS_OP_AND);
    assert(BUFSIZE % BLOCKLEN == 0);

    // Generates random numbers ad infinitum.
    char buff[BUFSIZE];
    for(;;){
        for(int i = 0; i<BUFSIZE; i++){
            buff[i] = randint(256);

            // Decide on each block
            if (i >= blocklen-1 && (i - blocklen + 1) % blocklen == 0){
                int reset_f = zero_chance <= 1 ? 1 : randint(zero_chance) == 0;
                if (reset_f){

                    // Evaluate term
                    int val = BITVAL(buff, i - blocklen + 1, bit_pos[0]);
                    for(int ctr=1; ctr<deg; ctr++){
                        if (bias_op == BIAS_OP_XOR){
                            val ^= BITVAL(buff, i - blocklen + 1, bit_pos[ctr]);
                        } else {
                            val &= BITVAL(buff, i - blocklen + 1, bit_pos[ctr]);
                        }
                    }

                    if (val == 0){
                        continue;
                    }

                    const int bitchange = bit_pos[randint(deg)];
                    const int byte_idx = i - blocklen + 1 + BITBYTE(bitchange);
                    if (bias_op == BIAS_OP_XOR){
                        buff[byte_idx] ^= BITMASK(bitchange);
                    } else {
                        buff[byte_idx] &= ~BITMASK(bitchange);
                    }
                }
            }
        }

        write(fileno(stdout), buff, BUFSIZE);
	    fflush(stdout);
    }

    return 0;
}
