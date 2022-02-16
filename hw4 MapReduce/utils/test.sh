make
g++ -o judge judge.cc
rm logdir/*.out
# srun -N2 -c5 ./mapreduce TEST01 7 3 /home/pp21/pp21s17/share/hw4/testcases/01.word 2 /home/pp21/pp21s17/share/hw4/testcases/01.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST01 7 /home/pp21/pp21s17/share/hw4/testcases/01.ans
# srun -N2 -c12 ./mapreduce TEST02 7 2 /home/pp21/pp21s17/share/hw4/testcases/02.word 2 /home/pp21/pp21s17/share/hw4/testcases/02.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST02 7 /home/pp21/pp21s17/share/hw4/testcases/02.ans
# srun -N3 -c10 ./mapreduce TEST03 10 2 /home/pp21/pp21s17/share/hw4/testcases/03.word 5 /home/pp21/pp21s17/share/hw4/testcases/03.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST03 10 /home/pp21/pp21s17/share/hw4/testcases/03.ans
# srun -N3 -c12 ./mapreduce TEST04 10 3 /home/pp21/pp21s17/share/hw4/testcases/04.word 5 /home/pp21/pp21s17/share/hw4/testcases/04.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST04 10 /home/pp21/pp21s17/share/hw4/testcases/04.ans
# srun -N3 -c12 ./mapreduce TEST05 10 1 /home/pp21/pp21s17/share/hw4/testcases/05.word 2 /home/pp21/pp21s17/share/hw4/testcases/05.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST05 10 /home/pp21/pp21s17/share/hw4/testcases/05.ans
# srun -N4 -c10 ./mapreduce TEST06 12 3 /home/pp21/pp21s17/share/hw4/testcases/06.word 3 /home/pp21/pp21s17/share/hw4/testcases/06.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST06 12 /home/pp21/pp21s17/share/hw4/testcases/06.ans
# srun -N4 -c12 ./mapreduce TEST07 13 3 /home/pp21/pp21s17/share/hw4/testcases/07.word 4 /home/pp21/pp21s17/share/hw4/testcases/07.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST07 13 /home/pp21/pp21s17/share/hw4/testcases/07.ans
srun -N4 -c12 ./mapreduce TEST08 12 3 /home/pp21/pp21s17/share/hw4/testcases/08.word 6 /home/pp21/pp21s17/share/hw4/testcases/08.loc /home/pp21/pp21s17/hw4/logdir/
./judge /home/pp21/pp21s17/hw4/logdir/TEST08 12 /home/pp21/pp21s17/share/hw4/testcases/08.ans
# srun -N4 -c12 ./mapreduce TEST09 7 3 /home/pp21/pp21s17/share/hw4/testcases/09.word 5 /home/pp21/pp21s17/share/hw4/testcases/09.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST09 7 /home/pp21/pp21s17/share/hw4/testcases/09.ans
# srun -N4 -c12 ./mapreduce TEST10 12 5 /home/pp21/pp21s17/share/hw4/testcases/10.word 10 /home/pp21/pp21s17/share/hw4/testcases/10.loc /home/pp21/pp21s17/hw4/logdir/
# ./judge /home/pp21/pp21s17/hw4/logdir/TEST10 12 /home/pp21/pp21s17/share/hw4/testcases/10.ans