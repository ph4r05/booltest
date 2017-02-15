CC = gcc -Wall -O3 -std=c99 -Wall

all:  rand

rand:  rand.c
	${CC} -o $@  rand.c ${LINKOPT}

.c.o:
	${CC} -c $<

clean:
	rm -rf *.o *~ *.dSYM rand
