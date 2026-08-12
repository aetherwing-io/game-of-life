/* life2.c — the traditional-pipeline equivalent of life2.hex, written for
 * the token/size comparison. Behavior-identical: 16x16 torus, R-pentomino,
 * ANSI clear-screen animation, 60ms frame delay, `life2 N` for N generations
 * (default/invalid -> 64). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 16

int main(int argc, char **argv) {
    long gens = argc > 1 ? strtol(argv[1], 0, 10) : 0;
    if (gens <= 0) gens = 64;
    unsigned char board[N * N] = {0}, next[N * N];
    int seed[][2] = {{8, 7}, {9, 7}, {7, 8}, {8, 8}, {8, 9}};
    for (int i = 0; i < 5; i++) board[seed[i][1] * N + seed[i][0]] = 1;
    struct timespec ts = {0, 60000000};
    while (gens--) {
        char out[8 + N * (N + 1) + 2];
        char *p = out + sprintf(out, "\x1b[2J\x1b[H");
        for (int i = 0; i < N * N; i++) {
            *p++ = board[i] ? '#' : '.';
            if (i % N == N - 1) *p++ = '\n';
        }
        *p++ = '\n';
        fwrite(out, 1, p - out, stdout);
        fflush(stdout);
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int n = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        if (dx || dy) n += board[((y + dy) & 15) * N + ((x + dx) & 15)];
                next[y * N + x] = n == 3 || (n == 2 && board[y * N + x]);
            }
        memcpy(board, next, sizeof board);
        nanosleep(&ts, 0);
    }
    return 0;
}
