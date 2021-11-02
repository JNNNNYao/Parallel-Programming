import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

proc = subprocess.Popen(['mpicxx', '-O3', '-lm', '-o', 'hw1_time', 'hw1_time.cc'])
outs, errs = proc.communicate()
single_node = True
if single_node:
    num_processes = [1, 4, 8, 12]
else:
    CORES_PER_NODE = 4
    num_node = [1, 1, 2, 3, 4]
    num_processes = [1, CORES_PER_NODE*1, CORES_PER_NODE*2, CORES_PER_NODE*3, CORES_PER_NODE*4]
IO_time, Comm_time, CPU_time, total_time = [], [], [], []
for i, num_proc in enumerate(num_processes):
    print('number of processes: {}'.format(num_proc))
    rounds = 3
    IO, Comm, CPU, total = 0.0, 0.0, 0.0, 0.0
    for j in range(rounds):
        if single_node:
            proc = subprocess.Popen(['srun', '-n{}'.format(num_proc), '-N1', './hw1_time', '536869888', './testcases/40.in', './output/40.out'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        else:
            proc = subprocess.Popen(['srun', '-n{}'.format(num_proc), '-N{}'.format(num_node[i]), '--mincpus={}'.format(math.ceil(num_proc/num_node[i])), './hw1_time', '536869888', './testcases/40.in', './output/40.out'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        outs, errs = proc.communicate()
        IO, Comm, CPU = IO+float(outs.split(' ')[0]), Comm+float(outs.split(' ')[1]), CPU+float(outs.split(' ')[2])
        total = total + float(outs.split(' ')[3])
    IO_time.append(IO/rounds)
    Comm_time.append(Comm/rounds)
    CPU_time.append(CPU/rounds)
    total_time.append(total/rounds)

IO_time, Comm_time, CPU_time, total_time = np.array(IO_time), np.array(Comm_time), np.array(CPU_time), np.array(total_time)

x = np.arange(len(num_processes))
plt.figure(1)
b1 = plt.bar(x, IO_time, color='teal', label='IO')
b2 = plt.bar(x, Comm_time, color='firebrick', label='Comm', bottom=IO_time)
b3 = plt.bar(x, CPU_time, color='yellowgreen', label='CPU', bottom=Comm_time + IO_time)
plt.xticks(x, num_processes)
if single_node:
    plt.xlabel('# of Processes (single node)')
else:
    plt.xlabel('# of Processes ({}cores/node)'.format(CORES_PER_NODE))
plt.ylabel('runtime (seconds)')
plt.legend(loc='upper right', shadow=True)
if single_node:
    plt.savefig('Time_profile_single.png')
else:
    plt.savefig('Time_profile_{}.png'.format(CORES_PER_NODE))

total_time = total_time[0] / total_time
plt.figure(2)
plt.plot(total_time, 's-', color = 'teal')
plt.xticks(x, num_processes)
if single_node:
    plt.xlabel('# of Processes (single node)')
else:
    plt.xlabel('# of Processes ({}cores/node)'.format(CORES_PER_NODE))
plt.ylabel('speedup')
if single_node:
    plt.savefig('Speedup_single.png')
else:
    plt.savefig('Speedup_{}.png'.format(CORES_PER_NODE))

plt.figure(3)
plt.plot(IO_time, 's-', color='teal', label='IO')
plt.plot(Comm_time, 's-', color='firebrick', label='Comm')
plt.plot(CPU_time, 's-', color='yellowgreen', label='CPU')
plt.xticks(x, num_processes)
if single_node:
    plt.xlabel('# of Processes (single node)')
else:
    plt.xlabel('# of Processes ({}cores/node)'.format(CORES_PER_NODE))
plt.ylabel('runtime (seconds)')
plt.legend(loc='upper right', shadow=True)
if single_node:
    plt.savefig('Time_profile_line_single.png')
else:
    plt.savefig('Time_profile_line_{}.png'.format(CORES_PER_NODE))

plt.figure(4)
IO_time = IO_time[0] / IO_time
Comm_time = Comm_time[0] / Comm_time
CPU_time = CPU_time[0] / CPU_time
plt.plot(IO_time, 's-', color='teal', label='IO')
plt.plot(Comm_time, 's-', color='firebrick', label='Comm')
plt.plot(CPU_time, 's-', color='yellowgreen', label='CPU')
plt.xticks(x, num_processes)
if single_node:
    plt.xlabel('# of Processes (single node)')
else:
    plt.xlabel('# of Processes ({}cores/node)'.format(CORES_PER_NODE))
plt.ylabel('speedup')
plt.legend(loc='upper left', shadow=True)
if single_node:
    plt.savefig('Speedup_each_single.png')
else:
    plt.savefig('Speedup_each_{}.png'.format(CORES_PER_NODE))
