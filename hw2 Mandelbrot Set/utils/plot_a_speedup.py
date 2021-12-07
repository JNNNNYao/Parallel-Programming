import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

proc = subprocess.Popen(['make', 'clean'])
outs, errs = proc.communicate()
proc = subprocess.Popen(['make'])
outs, errs = proc.communicate()
num_threads = [2, 4, 6, 8, 10, 12]
rounds = 3

# proc = subprocess.Popen(['srun', '-n1', '-c1', './hw2a_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
# outs, errs = proc.communicate()
# seq = float(outs.split(' ')[0])
# print("seq time: {}".format(seq))
seq = 114.547402

total_time = []
for i, num_t in enumerate(num_threads):
    print('number of thread: {}'.format(num_t))
    total = 0.0
    for j in range(rounds):
        proc = subprocess.Popen(['srun', '-n1', '-c{}'.format(num_t), './hw2a_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        outs, errs = proc.communicate()
        print(outs)
        total = total + float(outs.split(' ')[0])
        proc = subprocess.Popen(['hw2-diff', './testcases/strict21.png', 'out.png'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        outs, errs = proc.communicate()
        print(outs)
    total_time.append(total/rounds)
total_time = np.array(total_time)

proc = subprocess.Popen(['rm', 'out.png'])
outs, errs = proc.communicate()

x = np.arange(len(num_threads))

speedup = seq / total_time
plt.figure(1)
plt.plot(speedup, 's-', color = 'teal')
plt.plot(num_threads, '.--', color = 'olive')
plt.legend(['speedup', 'ideal speedup'])
plt.xticks(x, num_threads)
plt.xlabel('# of threads (single node)')
plt.ylabel('speedup')
plt.savefig('Speedup_a.png')
