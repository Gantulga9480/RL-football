import matplotlib.pyplot as plt

scores = []

with open("random_ball/first/rewards.txt", 'r') as f:
    data = f.readlines()
    scores = [float(item) for item in data]

scores_sum = []
window_size = 500

for i in range(len(scores) - window_size):
    scores_sum.append(sum(scores[i:i + window_size]) / window_size)

plt.plot(scores_sum)
plt.show()
