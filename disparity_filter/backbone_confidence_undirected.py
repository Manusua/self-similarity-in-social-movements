import numpy as np

NODOSMAX = 100000
NEDGESMAX = 1000000

internalnet = np.zeros((NEDGESMAX, 2), dtype=int)
weight = np.zeros(NEDGESMAX)
strenght = np.zeros(NODOSMAX)
ndegree = np.zeros(NODOSMAX, dtype=int)
nodepresent = np.zeros(NODOSMAX, dtype=int)
weightmin = 0.0

filename = '../Data/distancesbetweencities_Spain_ids.csv'  # input network
filenameout = '../Data/cities_Spain_stat.dat'  # output file

for i in range(NODOSMAX):
    strenght[i] = 0.0
    ndegree[i] = 0

for i in range(NEDGESMAX):
    weight[i] = 0.0

weighttotal = 0.0
NODOS = 0
nlink = 0

with open(filename, 'r') as f:
    for line in f:
        i, j, d = map(float, line.split(','))
        i, j = int(i), int(j)
        if d == 0:
            print(f"cities {i} {j} have distance 0")
            d += 0.1
        #w = 1.0 / d**(1.0)
        w = d
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

with open(filenameout, 'w') as f:
    for n in range(10000):
        wconfidence = float(n) * 0.0001
        nodepresent.fill(0)

        weightbackbone = 0.0
        nedgesbackbone = 0
        weightnodeszerodegree = 0.0

        for i in range(nlink):
            if (ndegree[internalnet[i, 1]] > 1) and (ndegree[internalnet[i, 0]] > 1):
                if (weight[i] / strenght[internalnet[i, 0]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 0]] - 1)))) or \
                   (weight[i] / strenght[internalnet[i, 1]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 1]] - 1)))):
                    weightbackbone += weight[i]
                    nedgesbackbone += 1
                    nodepresent[internalnet[i, 0]] = 1
                    nodepresent[internalnet[i, 1]] = 1
            elif (ndegree[internalnet[i, 1]] > 1) and (ndegree[internalnet[i, 0]] == 1):
                if (weight[i] / strenght[internalnet[i, 1]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 1]] - 1)))):
                    weightbackbone += weight[i]
                    nedgesbackbone += 1
                    nodepresent[internalnet[i, 0]] = 1
                    nodepresent[internalnet[i, 1]] = 1
            elif (ndegree[internalnet[i, 1]] == 1) and (ndegree[internalnet[i, 0]] > 1):
                if (weight[i] / strenght[internalnet[i, 0]] > (1 - (1 - wconfidence)**(1 / (ndegree[internalnet[i, 0]] - 1)))):
                    weightbackbone += weight[i]
                    nedgesbackbone += 1
                    nodepresent[internalnet[i, 0]] = 1
                    nodepresent[internalnet[i, 1]] = 1
            else:
                weightnodeszerodegree += weight[i]

        nodosbackbone = 0
        strenghtbackbone = 0.0
        for i in range(NODOS):
            if nodepresent[i] == 1:
                nodosbackbone += 1
                strenghtbackbone += strenght[i] - strenght[i]

        f.write(f"{wconfidence:.6f} {weightbackbone / weighttotal:.4f} {nodosbackbone / NODOS:.4f} {nedgesbackbone / nlink:.4f}\n")