import numpy as np
import matplotlib.pyplot as plt

# Coordinates for test points and illumination points
x = np.array([2, 3, 4, 5, 6, 7])
y = np.array([2, 3, 4, 5, 6, 7])

# Intensity matrix where rows correspond to test points and columns to illumination points
z = np.array([
    [1,    0.1, 0.05, 0,    0,    0   ],
    [0.1,  1,   0.1,  0.1,  0.05, 0.05],
    [0.05, 0.1, 1,    0.1,  0.05, 0.05],
    [0.05, 0.05,0.1,  1,    0.1,  0.05],
    [0.05, 0.05,0.05, 0.1,  1,    0.1 ],
    [0,    0,   0,    0.05, 0.1,  1   ]
])

# Create grid of x and y coordinates for each bar
X, Y = np.meshgrid(x, y)
X_flat = X.ravel()
Y_flat = Y.ravel()
Z_flat = z.ravel()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Width and depth for each bar in the waterfall plot
width = depth = 0.5
bottom = np.zeros_like(Z_flat)

ax.bar3d(X_flat, Y_flat, bottom, width, depth, Z_flat, shade=True)
ax.set_xlabel('Test Point')
ax.set_ylabel('Illumination Point')
ax.set_zlabel('Intensity')
ax.set_title('3D Waterfall Plot of Test Intensity')

plt.savefig('waterfall_plot.png')
print('Saved 3D waterfall plot as waterfall_plot.png')
