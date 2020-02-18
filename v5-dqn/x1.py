import numpy as np

a1 = [((1, 0, 1, 0, -1, 0, 0, 0, -1), -1),
     ((-1, 0, -1, 0, 1, 0, 0, 0, 1), 1),
     ((-1, 0, -1, 0, 1, 0, 0, 0, 1), -1)]

print(a1)
a2 = np.array(a1)
print(a2)
a3 = a1[:, 0]
print(a3)
a4 = a1[:, 1]
print(a4)

