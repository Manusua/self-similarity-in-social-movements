import numpy as np

NODOSMAX = 100000
NEDGESMAX = 1000000

internalnet = np.zeros((NEDGESMAX, 2), dtype=int)
weight = np.zeros(NEDGESMAX)
strenght = np.zeros(NODOSMAX)
ndegree = np.zeros(NODOSMAX, dtype=int)

weightmin = 0.0
wconfidence = 0.9977  # this is 1-alfa

filename = '../Data/distancesbetweencities_Spain_ids.csv'  # input network
filenameout = '../Data/backbone_Spain_0.9977_nw.net'  # output backbone

for i in range(NODOSMAX):
    strenght[i] = 0.0
    ndegree[i] = 0

for i in range(NEDGESMAX):
    weight[i] = 0.0

weighttotal = 0.0
NODOS = 0
nlink = 0

with open(filename, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        i, j, d = map(float, line.split())
        i, j = int(i), int(j)
        if d == 0:
            print(f"cities {i} {j} have distance 0")
            d += 0.1
        w = 1.0 / d**(1.0)
        nlink += 1
        internalnet[nlink - 1, 0] = i
        internalnet[nlink - 1, 1] = j
        strenght[i] += w
        strenght[j] += w
        ndegree[i] += 1
        ndegree[j] += 1
        weight[nlink - 1] = w
        weighttotal += w
        NODOS = max(NODOS, i, j)

weightbackbone = 0.0

with open(filenameout, 'w') as f:
    for i in range(nlink):
        if (ndegree[internalnet[i, 1]] > 1) and (ndegree[internalnet[i, 0]] > 1):
            if (weight[i] / strenght[internalnet[i, 0]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 0]] - 1)))) or \
               (weight[i] / strenght[internalnet[i, 1]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 1]] - 1)))):
                if weight[i] > weightmin:
                    f.write(f"{internalnet[i, 0]} {internalnet[i, 1]} {weight[i]}\n")
                    weightbackbone += weight[i]
        elif (ndegree[internalnet[i, 1]] > 1) and (ndegree[internalnet[i, 0]] == 1):
            if weight[i] / strenght[internalnet[i, 1]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 1]] - 1))):
                if weight[i] > weightmin:
                    f.write(f"{internalnet[i, 0]} {internalnet[i, 1]} {weight[i]}\n")
                    weightbackbone += weight[i]
        elif (ndegree[internalnet[i, 1]] == 1) and (ndegree[internalnet[i, 0]] > 1):
            if weight[i] / strenght[internalnet[i, 0]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 0]] - 1))):
                if weight[i] > weightmin:
                    f.write(f"{internalnet[i, 0]} {internalnet[i, 1]}\n")
                    weightbackbone += weight[i]

print('fraction of weight in backbone=', weightbackbone / weighttotal)