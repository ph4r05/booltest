
inp = '''
        R1(A, B, C, D, E, X[ 5],  8);
    R1(E, A, B, C, D, X[14],  9);
    R1(D, E, A, B, C, X[ 7],  9);
    R1(C, D, E, A, B, X[ 0], 11);
    R1(B, C, D, E, A, X[ 9], 13);
    R1(A, B, C, D, E, X[ 2], 15);
    R1(E, A, B, C, D, X[11], 15);
    R1(D, E, A, B, C, X[ 4],  5);
    R1(C, D, E, A, B, X[13],  7);
    R1(B, C, D, E, A, X[ 6],  7);
    R1(A, B, C, D, E, X[15],  8);
    R1(E, A, B, C, D, X[ 8], 11);
    R1(D, E, A, B, C, X[ 1], 14);
    R1(C, D, E, A, B, X[10], 14);
    R1(B, C, D, E, A, X[ 3], 12);
    R1(A, B, C, D, E, X[12],  6);

    R2(E, A, B, C, D, X[ 6],  9);
    R2(D, E, A, B, C, X[11], 13);
    R2(C, D, E, A, B, X[ 3], 15);
    R2(B, C, D, E, A, X[ 7],  7);
    R2(A, B, C, D, E, X[ 0], 12);
    R2(E, A, B, C, D, X[13],  8);
    R2(D, E, A, B, C, X[ 5],  9);
    R2(C, D, E, A, B, X[10], 11);
    R2(B, C, D, E, A, X[14],  7);
    R2(A, B, C, D, E, X[15],  7);
    R2(E, A, B, C, D, X[ 8], 12);
    R2(D, E, A, B, C, X[12],  7);
    R2(C, D, E, A, B, X[ 4],  6);
    R2(B, C, D, E, A, X[ 9], 15);
    R2(A, B, C, D, E, X[ 1], 13);
    R2(E, A, B, C, D, X[ 2], 11);

    R3(D, E, A, B, C, X[15],  9);
    R3(C, D, E, A, B, X[ 5],  7);
    R3(B, C, D, E, A, X[ 1], 15);
    R3(A, B, C, D, E, X[ 3], 11);
    R3(E, A, B, C, D, X[ 7],  8);
    R3(D, E, A, B, C, X[14],  6);
    R3(C, D, E, A, B, X[ 6],  6);
    R3(B, C, D, E, A, X[ 9], 14);
    R3(A, B, C, D, E, X[11], 12);
    R3(E, A, B, C, D, X[ 8], 13);
    R3(D, E, A, B, C, X[12],  5);
    R3(C, D, E, A, B, X[ 2], 14);
    R3(B, C, D, E, A, X[10], 13);
    R3(A, B, C, D, E, X[ 0], 13);
    R3(E, A, B, C, D, X[ 4],  7);
    R3(D, E, A, B, C, X[13],  5);

    R4(C, D, E, A, B, X[ 8], 15);
    R4(B, C, D, E, A, X[ 6],  5);
    R4(A, B, C, D, E, X[ 4],  8);
    R4(E, A, B, C, D, X[ 1], 11);
    R4(D, E, A, B, C, X[ 3], 14);
    R4(C, D, E, A, B, X[11], 14);
    R4(B, C, D, E, A, X[15],  6);
    R4(A, B, C, D, E, X[ 0], 14);
    R4(E, A, B, C, D, X[ 5],  6);
    R4(D, E, A, B, C, X[12],  9);
    R4(C, D, E, A, B, X[ 2], 12);
    R4(B, C, D, E, A, X[13],  9);
    R4(A, B, C, D, E, X[ 9], 12);
    R4(E, A, B, C, D, X[ 7],  5);
    R4(D, E, A, B, C, X[10], 15);
    R4(C, D, E, A, B, X[14],  8);

    R5(B, C, D, E, A, X[12] ,  8);
    R5(A, B, C, D, E, X[15] ,  5);
    R5(E, A, B, C, D, X[10] , 12);
    R5(D, E, A, B, C, X[ 4] ,  9);
    R5(C, D, E, A, B, X[ 1] , 12);
    R5(B, C, D, E, A, X[ 5] ,  5);
    R5(A, B, C, D, E, X[ 8] , 14);
    R5(E, A, B, C, D, X[ 7] ,  6);
    R5(D, E, A, B, C, X[ 6] ,  8);
    R5(C, D, E, A, B, X[ 2] , 13);
    R5(B, C, D, E, A, X[13] ,  6);
    R5(A, B, C, D, E, X[14] ,  5);
    R5(E, A, B, C, D, X[ 0] , 15);
    R5(D, E, A, B, C, X[ 3] , 13);
    R5(C, D, E, A, B, X[ 9] , 11);
    R5(B, C, D, E, A, X[11] , 11);'''

spc = 0
for idx, x in enumerate(inp.split('\n')):
    if x.strip() == '':
        spc += 1
        print(x)
        continue

    print('%s if (nr == %2d) goto finish_r;' % (x, idx + 1 - spc))

