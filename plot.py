import matplotlib.pyplot as plt
scores = []

with open("rewards_initial.txt", 'r') as f:
    data = f.readlines()
    scores = [float(item) for item in data]

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
