import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

proc = subprocess.Popen(['make', 'clean'])
outs, errs = proc.communicate()
proc = subprocess.Popen(['make'])
outs, errs = proc.communicate()
num_proc = [1, 2, 3, 4]
rounds = 3

# proc = subprocess.Popen(['srun', '-n1', '-c1', './hw2b_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
# outs, errs = proc.communicate()
# seq = float(outs.split(' ')[0]) + float(outs.split(' ')[1])
# print("seq time: {}".format(seq))
seq = 141.470585

cpu_time = []
comm_time = []
for i, num_p in enumerate(num_proc):
    print('number of process: {}'.format(num_p))
    cpu = 0.0
    comm = 0.0
    for j in range(rounds):
        proc = subprocess.Popen(['srun', '-n{}'.format(num_p), '-c6', '-N2', './hw2b_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        outs, errs = proc.communicate()
        print(outs)
        cpu = cpu + float(outs.split(' ')[0])
        comm = comm + float(outs.split(' ')[1])
        proc = subprocess.Popen(['hw2-diff', './testcases/strict21.png', 'out.png'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        outs, errs = proc.communicate()
        print(outs)
    cpu_time.append(cpu/rounds)
    comm_time.append(comm/rounds)
cpu_time = np.array(cpu_time)
comm_time = np.array(comm_time)

proc = subprocess.Popen(['rm', 'out.png'])
outs, errs = proc.communicate()

x = np.arange(len(num_proc))

speedup = seq / (cpu_time+comm_time)
plt.figure(1)
plt.plot(speedup, 's-', color = 'teal')
plt.plot(6*np.array(num_proc), '.--', color = 'olive')
plt.legend(['speedup', 'ideal speedup'])
plt.xticks(x, num_proc)
plt.xlabel('# of process (total 2 Node)')
plt.ylabel('speedup')
plt.savefig('Speedup_b.png')

plt.figure(2)
b1 = plt.bar(x, cpu_time, color='yellowgreen', label='CPU')
b2 = plt.bar(x, comm_time, color='firebrick', label='Comm', bottom=cpu_time)
plt.legend(loc='upper right', shadow=True)
plt.xticks(x, num_proc)
plt.xlabel('# of process (total 2 Node)')
plt.ylabel('runtime (seconds)')
plt.title('Time Profile')
plt.savefig('Time_profile_b.png')

plt.figure(3)
plt.bar(x, comm_time, color='firebrick', label='Comm')
low = min(comm_time)
high = max(comm_time)
plt.ylim([0, high+0.5*(high-low)])
plt.legend(loc='upper right', shadow=True)
plt.xticks(x, num_proc)
plt.xlabel('# of process (total 2 Node)')
plt.ylabel('runtime (seconds)')
plt.title('Time Profile of Communication time')
plt.savefig('Comm_profile_b.png')
