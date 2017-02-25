CC = gcc -Wall -O3 -std=c99 -Wall -lm
U01 = u01/ugfsr.c u01/util.c u01/unif01.c u01/addstr.c u01/num.c u01/chrono.c u01/bitset.c u01/usoft.c u01/ulcg.c
LINKOPT = -lm

all:  rand-tt400 rand-tt403 rand-tt775 rand-tt800 rand-t800 rand-toot73 rand-kirk81 rand-ripley90 rand-fushimi90 rand-ziff98

rand-tt400:
	${CC} -o $@ ${U01} rand.c -DTT400 ${LINKOPT}

rand-tt403:
	${CC} -o $@ ${U01} rand.c -DTT403 ${LINKOPT}

rand-tt775:
	${CC} -o $@ ${U01} rand.c -DTT775 ${LINKOPT}

rand-tt800:
	${CC} -o $@ ${U01} rand.c -DTT800 ${LINKOPT}

rand-t800:
	${CC} -o $@ ${U01} rand.c -DT800 ${LINKOPT}

rand-toot73:
	${CC} -o $@ ${U01} rand.c -DTOOT73 ${LINKOPT}

rand-kirk81:
	${CC} -o $@ ${U01} rand.c -DKIRK81 ${LINKOPT}

rand-ripley90:
	${CC} -o $@ ${U01} rand.c -DRIPLEY90 ${LINKOPT}

rand-fushimi90:
	${CC} -o $@ ${U01} rand.c -DFUSHIMI90 ${LINKOPT}

rand-ziff98:
	${CC} -o $@ ${U01} rand.c -DZIFF98 ${LINKOPT}


.c.o:
	${CC} -c $<

clean:
	rm -rf *.o *~ *.dSYM rand-*

