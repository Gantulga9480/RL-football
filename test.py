name = ""

with open("file.txt", 'r') as f:
    lines = f.readlines()

datas = []
for line in lines:
    idx = line.find('r:')
    data = line[idx + 3:-5]
    datas.append(float(data))

with open(name + "_rewards.txt", 'w') as f:
    f.writelines([f"{d}\n" for d in datas])
