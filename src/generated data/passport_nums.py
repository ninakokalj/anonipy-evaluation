import random
import numpy as np



nums = set()

while len(nums) < 300:

    passport_num = chr(random.choice([i for i in range(65, 91)]))
    if random.choice([1, 2]) == 2:
        passport_num += chr(random.choice([i for i in range(65, 91)]))

    while len(passport_num) < 9:
        passport_num += str(random.choice([i for i in range(0, 10)]))

    nums.add(passport_num)

np.savetxt("data/training/pass_nums.csv", list(nums), delimiter=',', fmt='%s')




