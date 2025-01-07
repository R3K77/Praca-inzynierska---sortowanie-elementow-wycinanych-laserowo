import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker



START = 0
END = 10.4

data = pd.read_csv('./Robot simulation/logData.csv')
time, x, y, z, theta = data['timeLog'], data['xLog'], data['yLog'], data['zLog'], data['thetaLog']
time = time - time[0]

plt.rc('font', family='Times New Roman', size=15)

fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)


axs[0].plot(time, x, label='X[m]', color='gray')
axs[0].set_ylabel('X [m]', font='times new roman')
axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
axs[0].grid(True)

axs[0].set_ylim(0.18, 0.42)

axs[1].plot(time, y, label='Y[m]', color='gray')
axs[1].set_ylabel('Y [m]', font='times new roman')
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[1].grid(True)

axs[2].plot(time, z, label='Z[m]', color='gray')
axs[2].set_ylabel('Z [m]', font='times new roman')
axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
axs[2].grid(True)

# axs[3].plot(time, theta + np.pi, label='theta', color='red')
# axs[3].set_ylabel('Theta [rad]', font='times new roman')
# axs[3].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
# axs[3].yaxis.set_major_formatter(lambda x, pos: f"{x/np.pi}π")
# axs[3].set_xlabel('Czas [s]', font='times new roman')
# # axs[3].legend()
# axs[3].grid(True)

for ax in axs:
    ax.set_xlim(START, END)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))


plt.tight_layout()
plt.show()

# 3D plot

# Subset the data based on time
mask = (time >= START) & (time <= END)
x_mask = x[mask]
y_mask = y[mask]
z_mask = z[mask]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', facecolor='white')
ax.plot(x_mask, y_mask, z_mask, label='Trajectory', color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.tight_layout()
plt.show()

# 3D plot with arrows

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, label='Trajectory', color='purple')

# # Add arrows to indicate direction
# for i in range(0, len(x)-1, max(1, len(x)//20)):
#     ax.quiver(x[i], y[i], z[i],
#               x[i+1]-x[i],
#               y[i+1]-y[i],
#               z[i+1]-z[i],
#               color='black', length=0.1, normalize=True)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.tight_layout()
# plt.show()