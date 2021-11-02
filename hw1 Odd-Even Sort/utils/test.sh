make clean
make
g++ -o test test.cc
# srun -n4 -N2 ./hw1 4 ./testcases/01.in ./output/01.out
# ./test 4 ./output/01.out ./testcases/01.out
# srun -n15 -N3 ./hw1 15 ./testcases/02.in ./output/02.out
# ./test 15 ./output/02.out ./testcases/02.out
# srun -n28 -N4 ./hw1 21 ./testcases/03.in ./output/03.out
# ./test 21 ./output/03.out ./testcases/03.out
# srun -n1 -N1 ./hw1 50 ./testcases/04.in ./output/04.out
# ./test 50 ./output/04.out ./testcases/04.out
# srun -n12 -N2 ./hw1 100 ./testcases/05.in ./output/05.out
# ./test 100 ./output/05.out ./testcases/05.out
# srun -n10 -N2 ./hw1 65536 ./testcases/06.in ./output/06.out
# ./test 65536 ./output/06.out ./testcases/06.out
# srun -n36 -N3 ./hw1 12345 ./testcases/07.in ./output/07.out
# ./test 12345 ./output/07.out ./testcases/07.out
# srun -n36 -N3 ./hw1 100000 ./testcases/08.in ./output/08.out
# ./test 100000 ./output/08.out ./testcases/08.out
# srun -n24 -N2 ./hw1 99999 ./testcases/09.in ./output/09.out
# ./test 99999 ./output/09.out ./testcases/09.out
# srun -n36 -N3 ./hw1 63942 ./testcases/10.in ./output/10.out
# ./test 63942 ./output/10.out ./testcases/10.out
# srun -n15 -N3 ./hw1 15 ./testcases/11.in ./output/11.out
# ./test 15 ./output/11.out ./testcases/11.out
# srun -n1 -N1 ./hw1 1 ./testcases/12.in ./output/12.out
# ./test 1 ./output/12.out ./testcases/12.out
# srun -n20 -N2 ./hw1 20 ./testcases/13.in ./output/13.out
# ./test 20 ./output/13.out ./testcases/13.out
# srun -n15 -N3 ./hw1 12345 ./testcases/14.in ./output/14.out
# ./test 12345 ./output/14.out ./testcases/14.out
# srun -n21 -N3 ./hw1 10059 ./testcases/15.in ./output/15.out
# ./test 10059 ./output/15.out ./testcases/15.out
# srun -n11 -N1 ./hw1 54923 ./testcases/16.in ./output/16.out
# ./test 54923 ./output/16.out ./testcases/16.out
# srun -n20 -N2 ./hw1 400000 ./testcases/17.in ./output/17.out
# ./test 400000 ./output/17.out ./testcases/17.out
# srun -n20 -N2 ./hw1 400000 ./testcases/18.in ./output/18.out
# ./test 400000 ./output/18.out ./testcases/18.out
# srun -n20 -N2 ./hw1 400000 ./testcases/19.in ./output/19.out
# ./test 400000 ./output/19.out ./testcases/19.out
# srun -n24 -N2 ./hw1 11183 ./testcases/20.in ./output/20.out
# ./test 11183 ./output/20.out ./testcases/20.out


srun -n24 -N2 ./hw1 12347 ./testcases/21.in ./output/21.out
./test 12347 ./output/21.out ./testcases/21.out
srun -n24 -N2 ./hw1 1000003 ./testcases/28.in ./output/28.out
./test 1000003 ./output/28.out ./testcases/28.out

# srun -n24 -N2 ./hw1 64123513 ./testcases/32.in ./output/32.out
# ./test 64123513 ./output/32.out ./testcases/32.out
# srun -n24 -N3 ./hw1 536870864 ./testcases/39.in ./output/39.out
# ./test 536870864 ./output/39.out ./testcases/39.out
# srun -n12 -N3 ./hw1 536869888 ./testcases/40.in ./output/40.out
# ./test 536869888 ./output/40.out ./testcases/40.out