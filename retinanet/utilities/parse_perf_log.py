#! /usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 2:
  print(f'usage: { sys.argv[0] } <path to the log>')
  exit()

without_mask = {
  'ap': [],
  'pre': [],
  'rec': [],
}

with_mask = {
  'ap': [],
  'pre': [],
  'rec': [],
}

incorrect = {
  'ap': [],
  'pre': [],
  'rec': [],
}

epoch_num = 1
line_num = 0

with open(sys.argv[1]) as f:
  for line in f:
    line_num += 1
    if line_num == 1 or line_num == 2 or line_num == 6 or line_num == 7 or line_num == 11 or line_num == 12:
      continue
    if line_num == 16:
      line_num = 1
      epoch_num += 1
      continue

    val = float(line.split()[1])
    if line_num == 3:
      without_mask['ap'].append(val)
    elif line_num == 4:
      without_mask['pre'].append(val)
    elif line_num == 5:
      without_mask['rec'].append(val)
    elif line_num == 8:
      with_mask['ap'].append(val)
    elif line_num == 9:
      with_mask['pre'].append(val)
    elif line_num == 10:
      with_mask['rec'].append(val)
    elif line_num == 13:
      incorrect['ap'].append(val)
    elif line_num == 14:
      incorrect['pre'].append(val)
    elif line_num == 15:
      incorrect['rec'].append(val)

last_epoch_p1 = epoch_num + 1
x_epoch = range(last_epoch_p1 - epoch_num, last_epoch_p1)

# for key, name in zip(['ap', 'pre', 'rec'], ['mAP', 'Precision', 'Recall']):
#   plt.figure(figsize=(15, 10))
#   plt.plot(x_epoch, without_mask[key], label=f'without_mask')
#   plt.plot(x_epoch, with_mask[key], label=f'with_mask')
#   plt.plot(x_epoch, incorrect[key], label=f'mask_worn_incorretly')
#   plt.ylabel(name)
#   plt.xlabel('Epoch')
#   plt.legend()
#   plt.show()

mAP = (np.array(without_mask['ap']) + np.array(with_mask['ap']) + np.array(incorrect['ap'])) / 3
plt.figure(figsize=(15, 10))
plt.plot(x_epoch, mAP)
plt.ylabel('mAP')
plt.xlabel('Epoch')
plt.show()
