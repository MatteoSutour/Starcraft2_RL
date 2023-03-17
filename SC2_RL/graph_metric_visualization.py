import os
import numpy as np
import matplotlib.pyplot as plt

with open(os.path.join(os.path.dirname(__file__), "results.txt"), "r") as f:
    enemies_life_end_episode = [float(line.split()[1]) for line in f.readlines()]

enemies_life_end_episode = np.mean(np.array(enemies_life_end_episode[:-(len(enemies_life_end_episode)%10)]).reshape(-1, 10), axis=1)

plt.figure()
plt.plot(10*np.arange(len(enemies_life_end_episode)), enemies_life_end_episode)
plt.title("Mean total remaining enemies' health and shield at the end of each episode")
plt.xlabel("Episodes")
plt.ylabel("Total health + shield")
plt.show()