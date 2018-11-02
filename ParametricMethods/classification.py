import numpy as np
from matplotlib import pyplot as plt
import math

n1 = 20
n2 = 50

m1 = -2.6
m2 = 3.3

std1 = 2
std2 = 3

data = []

d1 = []
for i in range(n1):
    x1 = np.random.normal(m1, std1)

    d1.append([x1, 0])

d1.sort()
data += d1

d2 = []
for i in range(n2):
    x2 = np.random.normal(m2, std2)

    d2.append([x2, 1])

d2.sort()
data += d2

data = np.array(data)


# MLE Estimates
em1 = 0
em2 = 0

for x, c in data:
    if c == 0:
        em1 += x
    else:
        em2 += x

em1 /= n1
em2 /= n2

evar1 = 0
evar2 = 0

for x, c in data:
    if c == 0:
        evar1 += (x - em1) ** 2
    else:
        evar2 += (x - em2) ** 2

evar1 /= n1
evar2 /= n2

print("MLE mean1 = %s, real mean1 = %s" % (em1, m1))
print("MLE mean2 = %s, real mean2 = %s" % (em2, m2))

print("MLE var1 = %s, real var1 = %s" % (evar1, std1 ** 2))
print("MLE var2 = %s, real var2 = %s" % (evar2, std2 ** 2))

# Priors
pc1 = n1 / (n1 + n2)
pc2 = n2 / (n1 + n2)

points = [p for p in np.arange(-10, 10, 0.1)]

norm_density = lambda point, m, var: 1 / (math.sqrt(var) * np.sqrt(2 * np.pi)) * np.exp(-(point - m) ** 2 / (2 * math.sqrt(var) ** 2))

plt.plot(points, [norm_density(p, em1, evar1) for p in points], color='r')
plt.plot(points, [norm_density(p, em2, evar2) for p in points], color='b')

posterior = lambda point, likelihood, prior: (likelihood * prior) / (norm_density(point, em1, evar1) * pc1 + norm_density(point, em2, evar2) * pc2)

plt.plot(points, [posterior(p, norm_density(p, em1, evar1), pc1) for p in points], color='y')
plt.plot(points, [posterior(p, norm_density(p, em2, evar2), pc2) for p in points], color='g')



plt.show()