import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

proc = subprocess.Popen(['make', 'clean'])
outs, errs = proc.communicate()
proc = subprocess.Popen(['make'])
outs, errs = proc.communicate()
num_threads = 8

print('number of thread: {}'.format(num_threads))  
proc = subprocess.Popen(['srun', '-n1', '-c{}'.format(num_threads), './hw2a_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
loading = outs.split(' ')[1:1+num_threads]
loading = np.array(loading)
loading = loading.astype(np.float32)
print(loading)
proc = subprocess.Popen(['hw2-diff', './testcases/strict21.png', 'out.png'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
print(outs)

proc = subprocess.Popen(['rm', 'out.png'])
outs, errs = proc.communicate()

x = np.arange(num_threads)
plt.figure(1)
plt.bar(x, loading, color='teal', label='loading')
low = min(loading)
high = max(loading)
plt.ylim([low-0.5*(high-low), high+0.5*(high-low)])
plt.xlabel('thread ID')
plt.ylabel('runtime (seconds)')
loading_std = np.std(loading)
plt.title('Standard Deviation: {}\nMax-Min: {}'.format(loading_std, high-low))
plt.savefig('Load_balance_{}_1.png'.format(num_threads))

print('number of thread: {}'.format(num_threads))  
proc = subprocess.Popen(['srun', '-n1', '-c{}'.format(num_threads), './hw2a_time', 'out.png', '10000', '-1.55555', '-1.55515', '-0.0002', '0.0002', '6923', '6923'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
loading = outs.split(' ')[1:1+num_threads]
loading = np.array(loading)
loading = loading.astype(np.float32)
print(loading)
proc = subprocess.Popen(['hw2-diff', './testcases/strict21.png', 'out.png'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
outs, errs = proc.communicate()
print(outs)

proc = subprocess.Popen(['rm', 'out.png'])
outs, errs = proc.communicate()

x = np.arange(num_threads)
plt.figure(2)
plt.bar(x, loading, color='teal', label='loading')
low = min(loading)
high = max(loading)
plt.ylim([low-0.5*(high-low), high+0.5*(high-low)])
plt.xlabel('thread ID')
plt.ylabel('runtime (seconds)')
loading_std = np.std(loading)
plt.title('Standard Deviation: {}\nMax-Min: {}'.format(loading_std, high-low))
plt.savefig('Load_balance_{}_2.png'.format(num_threads))
