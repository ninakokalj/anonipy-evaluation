# 123-45-6789
# 987-65-4321
# 123-45-6789
# 987-65-4321

import random
import numpy as np

def generate_tax_ids(count):

    ids = set()
    while len(ids) < count:
        first = random.choice([i for i in range(100, 1000)])
        second = random.choice([i for i in range(10, 100)])
        third = random.choice([i for i in range(1000, 10000)])
        ids.add(f"{first}-{second}-{third}")
    
    np.savetxt("data/training/tax_ids.csv", list(ids), delimiter=',', fmt='%s')


generate_tax_ids(300)