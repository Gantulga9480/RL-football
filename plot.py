import numpy as np
import matplotlib
import matplotlib.pylab as plt
import argparse

# font = {'weight': 'normal', 'size': 12}
# matplotlib.rc('font', **font)
plt.style.use('seaborn-v0_8')
# plt.style.use('science')

parser = argparse.ArgumentParser()
parser.add_argument('rewards', nargs='+')
parser.add_argument('--width', type=int, default=1000)
parser.add_argument("--average", action="store_true")
parser.add_argument('--save', action="store_true")
args = parser.parse_args()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if not args.average:
    rewards_list = []
    file_names = []
    for file in args.rewards:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                rewards_list.append(np.array([float(item.strip()) for item in lines]))
                file_names.append(file.split(".")[0].split('\\')[-1])
        except IsADirectoryError:
            continue

    avgs = []
    maxs = []
    mins = []
    for rewards in rewards_list:
        avgs.append(moving_average(rewards, args.width))
        imax = []
        imin = []
        for i in range(len(rewards) - args.width + 1):
            imax.append(np.max(rewards[i:i + args.width]))
            imin.append(np.min(rewards[i:i + args.width]))
        maxs.append(imax)
        mins.append(imin)

    fig1, ax1 = plt.subplots()
    fig1.set_figwidth(18.75)
    fig1.set_figheight(12.5)
    fig1.tight_layout(h_pad=1)
    for i, r in enumerate(avgs):
        x = np.arange(0, len(r), 1)
        # ax1.plot(x, r, label=file_names[i], linewidth="2", c='#55a868')
        ax1.fill_between(x, y1=mins[i], y2=maxs[i], alpha=0.3)
        ax1.plot(x, r, label=file_names[i])
        ax1.axhline(0, c='r')
    plt.title("Training curve")
    ax1.legend()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    plt.show()

elif args.average:
    rewards = []
    file_names = []
    for file in args.rewards:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                data = [float(item.strip()) for item in lines]
                rewards.append(np.array(data))
                file_names.append(file.split(".")[0])
        except IsADirectoryError:
            continue
    np_r = np.array(rewards)
    np_r = np_r.mean(axis=0)
    avg = moving_average(np_r, args.width)

    fig1, ax1 = plt.subplots()
    ax1.plot(avg, label="Average", linewidth="5")
    ax1.legend()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    plt.title("Average training curve")
    plt.show()

    if args.save:
        with open(f'{file_names[0]}', 'w') as f:
            f.writelines([str(item) + '\n' for item in avg])
