import matplotlib.pyplot as plt

scores = []

with open('rewards.txt', 'r') as f:
    data = f.readlines()
    scores = [float(item) for item in data]

scores_sum = []
window_size = 100

for i in range(len(scores)):
    scores_sum.append(sum(scores[i:i + window_size]) / window_size)

plt.plot(scores_sum)
plt.show()
