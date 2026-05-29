import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# Physical Constants
# ==============================
G = 6.67430e-11
c = 3e8
M_sun = 1.989e30

# ==============================
# Lens Parameters
# ==============================
M = 5e36                     # Supermassive lens mass
D_L = 1e22
D_S = 2e22
D_LS = D_S - D_L

# Einstein Radius
theta_E = np.sqrt(
    (4 * G * M / c**2) * (D_LS / (D_L * D_S))
)

print("Einstein Radius:", theta_E)

# ==============================
# Create Grid
# ==============================
N = 400
x = np.linspace(-4e21, 4e21, N)
y = np.linspace(-4e21, 4e21, N)
X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
R[R == 0] = 1e10  # Avoid division by zero

# ==============================
# Deflection Angle
# ==============================
alpha = (4 * G * M) / (c**2 * R)

# Deflected coordinates
X_def = X - alpha * (X / R)
Y_def = Y - alpha * (Y / R)

# ==============================
# Magnification
# ==============================
u = R / theta_E

magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
magnification = np.clip(magnification, 1, 50)

# ==============================
# Background Stars
# ==============================
np.random.seed(42)

num_stars = 800

stars_x = np.random.uniform(-4e21, 4e21, num_stars)
stars_y = np.random.uniform(-4e21, 4e21, num_stars)

# Lens distortion on stars
stars_r = np.sqrt(stars_x**2 + stars_y**2)
stars_r[stars_r == 0] = 1e10

star_alpha = (4 * G * M) / (c**2 * stars_r)

stars_x_def = stars_x - star_alpha * (stars_x / stars_r)
stars_y_def = stars_y - star_alpha * (stars_y / stars_r)

# ==============================
# Figure Setup
# ==============================
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("black")

# ------------------------------
# Magnification Plot
# ------------------------------
ax1 = plt.subplot(1, 2, 1)
ax1.set_facecolor("black")

img1 = ax1.imshow(
    magnification,
    extent=(-4e21, 4e21, -4e21, 4e21),
    origin='lower',
    cmap='inferno'
)

ax1.scatter(stars_x_def, stars_y_def,
            s=1,
            color='white',
            alpha=0.8)

# Einstein Ring
ring = plt.Circle(
    (0, 0),
    theta_E * 1e21,
    color='cyan',
    fill=False,
    linewidth=1.5
)

ax1.add_artist(ring)

# Black Hole
blackhole = plt.Circle(
    (0, 0),
    2e20,
    color='black'
)

ax1.add_artist(blackhole)

ax1.set_title(
    "Gravitational Lensing Magnification",
    color='white',
    fontsize=14
)

ax1.set_xlabel("X (m)", color='white')
ax1.set_ylabel("Y (m)", color='white')

ax1.tick_params(colors='white')

cbar1 = plt.colorbar(img1, ax=ax1)
cbar1.set_label("Magnification", color='white')
cbar1.ax.yaxis.set_tick_params(color='white')

# ------------------------------
# Deflection Plot
# ------------------------------
ax2 = plt.subplot(1, 2, 2)
ax2.set_facecolor("black")

img2 = ax2.imshow(
    alpha,
    extent=(-4e21, 4e21, -4e21, 4e21),
    origin='lower',
    cmap='plasma'
)

# Space-time Grid
for gx in np.linspace(-4e21, 4e21, 20):
    ax2.plot(
        np.full_like(y, gx),
        y,
        color='white',
        alpha=0.08
    )

for gy in np.linspace(-4e21, 4e21, 20):
    ax2.plot(
        x,
        np.full_like(x, gy),
        color='white',
        alpha=0.08
    )

# Curved Light Rays
for angle in np.linspace(0, 2*np.pi, 25):
    r_line = np.linspace(5e20, 4e21, 500)

    ray_x = r_line * np.cos(angle)
    ray_y = r_line * np.sin(angle)

    ray_r = np.sqrt(ray_x**2 + ray_y**2)

    bend = (4 * G * M) / (c**2 * ray_r)

    ray_x_def = ray_x - bend * np.cos(angle)
    ray_y_def = ray_y - bend * np.sin(angle)

    ax2.plot(
        ray_x_def,
        ray_y_def,
        color='cyan',
        linewidth=0.5,
        alpha=0.7
    )

# Black Hole Center
ax2.add_artist(
    plt.Circle((0, 0), 2e20, color='black')
)

ax2.set_title(
    "Space-Time Deflection",
    color='white',
    fontsize=14
)

ax2.set_xlabel("X (m)", color='white')
ax2.set_ylabel("Y (m)", color='white')

ax2.tick_params(colors='white')

cbar2 = plt.colorbar(img2, ax=ax2)
cbar2.set_label("Deflection", color='white')
cbar2.ax.yaxis.set_tick_params(color='white')

# ==============================
# Animation
# ==============================
def animate(frame):

    shift = np.sin(frame / 15) * 2e20

    ring.set_radius(theta_E * 1e21 + shift)

    return ring,

ani = FuncAnimation(
    fig,
    animate,
    frames=200,
    interval=50,
    blit=True
)

plt.tight_layout()
plt.show()