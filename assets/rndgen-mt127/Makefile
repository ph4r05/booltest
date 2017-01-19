CC = gcc -Wall -O3 -std=c99 -Wall

all:  rand

rand:  rand.c
	${CC} -o $@ tinymt32.c rand.c ${LINKOPT}

.c.o:
	${CC} -c $<

clean:
	rm -rf *.o *~ *.dSYM rand

