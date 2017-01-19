CC = g++ -Wall -O3 -std=c++11 -Wall

all:  rand

rand:  rand.cpp
	${CC} -o $@ rand.cpp ${LINKOPT}

.c.o:
	${CC} -c $<

clean:
	rm -rf *.o *~ *.dSYM rand

