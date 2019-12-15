import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

fls = ["multitask005.log", "multitask02.log", "multitask05.log"]
all_cont = []

for fl in fls:
    with open(fl, "r") as f:
        all_cont.append(f.readlines())

for content in all_cont:
    val_dice = []
    acc = []
    for line in content:
        line = line.strip()

        if "Dice:" in line:
            val_dice.append(float(line.split()[1]))
        if "Err:" in line:
            stri = line.split("[")[1]
            stri = stri.strip()[0:-1]
            stri = [float(x) for x in stri.split()]
            acc.append(np.mean(stri))

    print(np.mean(val_dice[:-5]))
    print(np.max(val_dice))
    print(np.mean(acc[:-5]))
    print(np.min(acc))


    
