#! /usr/bin/env python3
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
  print(f'usage: { sys.argv[0] } <path to the log>')
  exit()

prev_epoch = -1
epoch_num = 0
classify_loss_list = []
regress_loss_list = []
# running_loss_list = []

classify_loss = 0.0
regress_loss = 0.0
# running_loss = 0.0
count = 0

with open(sys.argv[1]) as f:
  for line in f:
    if "Epoch" in line:
      # example: Epoch: 99 | Iteration: 298 | Classification loss: 0.00001 | Regression loss: 0.00413 | Running loss: 0.00686
      arr = line.split(' ')
      epoch = arr[1]

      if epoch != prev_epoch:
        if prev_epoch != -1:
          classify_loss_list.append(classify_loss / count)
          regress_loss_list.append(regress_loss / count)
          # running_loss_list.append(running_loss / count)

        prev_epoch = epoch
        classify_loss = 0.0
        regress_loss = 0.0
        # running_loss = 0.0
        count = 0
        epoch_num += 1

      classify_loss += float(arr[8])
      regress_loss += float(arr[12])
      # running_loss += float(arr[16])
      count += 1

classify_loss_list.append(classify_loss / count)
regress_loss_list.append(regress_loss / count)
# running_loss_list.append(running_loss / count)

last_epoch_p1 = int(prev_epoch) + 1
x_epoch = range(last_epoch_p1 - epoch_num, last_epoch_p1)

plt.figure(figsize=(15, 10))
plt.plot(x_epoch, classify_loss_list, label='classification_loss')
plt.plot(x_epoch, regress_loss_list, label='regression_loss')
# plt.plot(x_epoch, running_loss_list, label='run_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
