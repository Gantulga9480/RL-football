import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='')
args = parser.parse_args()
scores = []
if args.file:
    with open(args.file, 'r') as f:
        data = f.readlines()
        scores = [float(item) for item in data]
else:
    quit()

scores_sum_small = []
scores_sum_big = []

window_size = 50
for i in range(len(scores) - window_size):
    scores_sum_small.append(sum(scores[i:i + window_size]) / window_size)

window_size = 500
for i in range(len(scores) - window_size):
    scores_sum_big.append(sum(scores[i:i + window_size]) / window_size)

fig, ax1 = plt.subplots()
ax1.plot(scores_sum_small, alpha=0.4, c='r', linewidth='10')
ax2 = ax1.twinx()
ax2.plot(scores_sum_big, c='b', linewidth='4')
plt.show()
