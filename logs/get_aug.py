import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

fls = ["out_Aug0.log", "out_Aug1.log", "out_Aug2.log"]
all_cont = []

for fl in fls:
    with open(fl, "r") as f:
        all_cont.append(f.readlines())

for content in all_cont:
    val_dice = []
    for line in content:
        line = line.strip()

        if "Dice:" in line:
            val_dice.append(float(line.split()[1]))

    print(np.mean(val_dice[:-10]))
    print(np.max(val_dice))


    indices = list(range(len(val_dice)))

    plt.plot(indices, val_dice)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")

plt.legend(['No aug', "LR/UD", "LR"])
plt.savefig("augmentation.png")


    
