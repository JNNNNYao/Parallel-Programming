make
srun -p prof -N1 -n1 --gres=gpu:1 nvprof --kernels cal_phase3 --metrics shared_load_throughput,shared_store_throughput ./hw3-2 /home/pp21/pp21s17/share/hw3-2/cases/c20.1 out.out 
rm *.out