from tqdm import tqdm
import numpy as np
from collections import deque

N = 512

# l = []

# for i in tqdm(range(10000)):
#     a = np.arange(N**3).reshape(N, N, N)
#     l.append(a)
#     if i > 50:
#         l.pop(0)


d = deque()

for i in tqdm(range(10000)):
    a = np.arange(N**3).reshape(N, N, N)
    d.append(a)
    if i > 50:
        d.popleft()