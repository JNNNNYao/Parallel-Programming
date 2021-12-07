import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

proc = subprocess.Popen(['make', 'clean'])
outs, errs = proc.communicate()
proc = subprocess.Popen(['make'])
outs, errs = proc.communicate()
num_proc = 4

print('number of process: {}'.format(num_proc))
proc = subprocess.Popen(['srun', '-n{}'.format(num_proc), '-c6', '-N2', './hw2b_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
print(outs)
loading = outs.split(' ')[2:]
loading = np.array(loading)
loading = loading.astype(np.float32)
print(loading)
proc = subprocess.Popen(['hw2-diff', './testcases/strict21.png', 'out.png'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
print(outs)

proc = subprocess.Popen(['rm', 'out.png'])
outs, errs = proc.communicate()

x = np.arange(num_proc)
plt.figure(1)
plt.bar(x, loading, color='teal', label='loading')
plt.xticks(x)
low = min(loading)
high = max(loading)
plt.ylim([6.44, 6.52])
plt.xlabel('process rank ID (total 2 Node)')
plt.ylabel('runtime (seconds)')
loading_std = np.std(loading)
plt.title('Standard Deviation: {}\nMax-Min: {}'.format(loading_std, high-low))
plt.savefig('Load_balance_b_{}_1.png'.format(num_proc))

print('number of process: {}'.format(num_proc))
proc = subprocess.Popen(['srun', '-n{}'.format(num_proc), '-c6', '-N2', './hw2b_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
loading = outs.split(' ')[2:]
loading = np.array(loading)
loading = loading.astype(np.float32)
print(loading)
proc = subprocess.Popen(['hw2-diff', './testcases/strict21.png', 'out.png'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
print(outs)

proc = subprocess.Popen(['rm', 'out.png'])
outs, errs = proc.communicate()

x = np.arange(num_proc)
plt.figure(1)
plt.bar(x, loading, color='teal', label='loading')
low = min(loading)
high = max(loading)
plt.ylim([6.44, 6.52])
plt.xlabel('process rank ID (total 2 Node)')
plt.ylabel('runtime (seconds)')
loading_std = np.std(loading)
plt.title('Standard Deviation: {}\nMax-Min: {}'.format(loading_std, high-low))
plt.savefig('Load_balance_b_{}_2.png'.format(num_proc))

