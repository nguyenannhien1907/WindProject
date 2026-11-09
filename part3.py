import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. THONG SO DAU VAO THEO TCVN 2737:2023
# ==========================================
num_floors = 30
h_floor = 3.5
B = 30.0

W0 = 1.25  # kPa - vung IV (ven bien)
alpha_terrain = 0.16  # dia hinh B
zg = 300.0

floors = np.arange(1, num_floors + 1)
z = floors * h_floor

k_z = 2.01 * (np.maximum(z, 3.0) / zg) ** (2 * alpha_terrain)

c_square = 1.4
c_chamfer = 1.1
c_round = 0.95

p_square = W0 * k_z * c_square
p_chamfer = W0 * k_z * c_chamfer
p_round = W0 * k_z * c_round

A_floor = B * h_floor
F_square = p_square * A_floor
F_chamfer = p_chamfer * A_floor
F_round = p_round * A_floor

k_story = 3.5e5
K = np.zeros((num_floors, num_floors))
for i in range(num_floors):
    if i == num_floors - 1:
        K[i, i] = k_story
    else:
        K[i, i] = 2 * k_story
        K[i, i + 1] = -k_story
        K[i + 1, i] = -k_story

u_square = np.linalg.solve(K, F_square) * 1000
u_chamfer = np.linalg.solve(K, F_chamfer) * 1000
u_round = np.linalg.solve(K, F_round) * 1000

RED = "#C0392B"
BLUE = "#1C7293"
GREEN = "#2C5F2D"

plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#444444"})

# ---- Figure 1: wind pressure profile ----
fig1, ax1 = plt.subplots(figsize=(6.2, 6.2))
ax1.plot(p_square, z, "-o", color=RED, markersize=4, linewidth=2.2, label="Vuông sắc cạnh (c=1.4)")
ax1.plot(p_chamfer, z, "-s", color=BLUE, markersize=4, linewidth=2.2, label="Vát góc (c=1.1)")
ax1.plot(p_round, z, "-^", color=GREEN, markersize=4, linewidth=2.2, label="Bo tròn (c=0.95)")
ax1.set_title("Áp lực gió theo chiều cao\n(TCVN 2737:2023)", fontweight="bold")
ax1.set_xlabel("Áp lực gió W(z) [kPa]")
ax1.set_ylabel("Độ cao z [m]")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left", fontsize=10, framealpha=0.95)
fig1.tight_layout()
fig1.savefig("fig_wind_pressure.png", dpi=200, transparent=True)

# ---- Figure 2: displacement comparison ----
fig2, ax2 = plt.subplots(figsize=(6.2, 6.2))
ax2.plot(u_square, z, "-o", color=RED, markersize=4, linewidth=2.2, label="Vuông (gốc)")
ax2.plot(u_chamfer, z, "-s", color=BLUE, markersize=4, linewidth=2.2, label="Vát góc (-21%)")
ax2.plot(u_round, z, "-^", color=GREEN, markersize=4, linewidth=2.2, label="Bo tròn (-32%)")
ax2.set_title("Chuyển vị ngang công trình\n(mô hình Shear Building)", fontweight="bold")
ax2.set_xlabel("Chuyển vị ngang u(z) [mm]")
ax2.set_ylabel("Độ cao z [m]")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)
fig2.tight_layout()
fig2.savefig("fig_displacement.png", dpi=200, transparent=True)

# ---- Figure 3: bar chart summary (base shear + top displacement) ----
labels = ["Vuông\n(gốc)", "Vát góc", "Bo tròn"]
top_disp = [u_square[-1], u_chamfer[-1], u_round[-1]]
base_shear = [np.sum(F_square), np.sum(F_chamfer), np.sum(F_round)]

fig3, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 4.6))
colors = [RED, BLUE, GREEN]
bars1 = axa.bar(labels, top_disp, color=colors, width=0.55)
axa.set_title("Chuyển vị đỉnh (mm)", fontweight="bold")
axa.grid(True, axis="y", linestyle="--", alpha=0.4)
for b, v in zip(bars1, top_disp):
    axa.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")

bars2 = axb.bar(labels, base_shear, color=colors, width=0.55)
axb.set_title("Lực cắt đáy (kN)", fontweight="bold")
axb.grid(True, axis="y", linestyle="--", alpha=0.4)
for b, v in zip(bars2, base_shear):
    axb.text(b.get_x() + b.get_width() / 2, v + 20, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")

fig3.tight_layout()
fig3.savefig("fig_summary_bars.png", dpi=200, transparent=True)

print("Top displacement (mm):", top_disp)
print("Base shear (kN):", base_shear)
print("Giam % vat goc:", 100 * (1 - top_disp[1] / top_disp[0]))
print("Giam % bo tron:", 100 * (1 - top_disp[2] / top_disp[0]))
print("DONE")

# ==========================================
# MODULE 9: MO PHONG 3D CA 3 TOA NHA (matplotlib)
# ==========================================
# Ve 3 toa nha (vuong / vat goc / bo tron) trong khong gian 3D, to mau tung
# tang theo ap luc gio W(z), va lac nhe theo chuyen vi ngang u(z) da giai o
# Module 2. Khi chay binh thuong (khong dung backend "Agg"), cua so hien ra
# co the XOAY bang chuot trai, PHONG TO bang chuot cuon.

import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def squircle_footprint(hw, p=3.0, n=48):
    """Bien dang 'bo tron' - giua hinh vuong va hinh tron (sieu ellipse)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    denom = (np.abs(ct) ** p + np.abs(st) ** p) ** (1.0 / p)
    return list(zip(hw * ct / denom, hw * st / denom))


def square_footprint(hw):
    return [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]


def chamfer_footprint(hw, cut=0.16):
    c = 2 * hw * cut
    return [
        (-hw + c, -hw), (hw - c, -hw), (hw, -hw + c), (hw, hw - c),
        (hw - c, hw), (-hw + c, hw), (-hw, hw - c), (-hw, -hw + c),
    ]


def interp_u(z_query, z_arr, u_arr):
    """Noi suy chuyen vi ngang u tai do cao bat ky (mong = 0 o chan cong trinh)."""
    z_full = np.concatenate(([0.0], z_arr))
    u_full = np.concatenate(([0.0], u_arr))
    return np.interp(z_query, z_full, u_full)


def build_building(ax, footprint, z_arr, p_arr, u_arr, x_offset, norm, cmap, sway_scale=4.0):
    """Dung tung tang thanh 1 khoi hop 3D, to mau theo ap luc gio, lech theo u(z)."""
    n = len(footprint)
    for i in range(len(z_arr)):
        z_bot = 0.0 if i == 0 else z_arr[i - 1]
        z_top = z_arr[i]
        # chuyen vi ngang (mm -> m), nhan sway_scale de nhin ro hon tren hinh ve
        dx_bot = interp_u(z_bot, z_arr, u_arr) / 1000.0 * sway_scale
        dx_top = interp_u(z_top, z_arr, u_arr) / 1000.0 * sway_scale

        verts_bot = [(px + x_offset + dx_bot, py, z_bot) for px, py in footprint]
        verts_top = [(px + x_offset + dx_top, py, z_top) for px, py in footprint]
        color = cmap(norm(p_arr[i]))

        faces = []
        for j in range(n):
            j2 = (j + 1) % n
            faces.append([verts_bot[j], verts_bot[j2], verts_top[j2], verts_top[j]])
        ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor="white", linewidths=0.15, alpha=0.95))

    # mai nha (mat tren cung)
    z_roof = z_arr[-1]
    dx_roof = interp_u(z_roof, z_arr, u_arr) / 1000.0 * sway_scale
    roof = [(px + x_offset + dx_roof, py, z_roof) for px, py in footprint]
    ax.add_collection3d(Poly3DCollection([roof], facecolor=cmap(norm(p_arr[-1])), edgecolor="white", linewidths=0.3))


hw = 4.75  # nua be rong mat bang (m) - be rong 9.5m cho ca 3 phuong an
spacing = 16.0  # khoang cach giua tam 3 toa nha (m)

all_p = np.concatenate([p_square, p_chamfer, p_round])
norm = Normalize(vmin=all_p.min(), vmax=all_p.max())
try:
    cmap = matplotlib.colormaps["RdYlBu_r"]  # matplotlib >= 3.7
except AttributeError:
    cmap = cm.get_cmap("RdYlBu_r")  # matplotlib cu hon

fig3d = plt.figure(figsize=(11, 7))
ax3d = fig3d.add_subplot(111, projection="3d")

build_building(ax3d, square_footprint(hw), z, p_square, u_square, -spacing, norm, cmap)
build_building(ax3d, chamfer_footprint(hw), z, p_chamfer, u_chamfer, 0.0, norm, cmap)
build_building(ax3d, squircle_footprint(hw), z, p_round, u_round, spacing, norm, cmap)

ax3d.set_xlim(-spacing - hw - 2, spacing + hw + 2)
ax3d.set_ylim(-hw - 10, hw + 10)
ax3d.set_zlim(0, z[-1] * 1.05)
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Độ cao Z (m)")
ax3d.set_title("Mô phỏng 3D — Vuông · Vát góc · Bo tròn\n(màu = áp lực gió W(z), lệch = chuyển vị ngang u(z) phóng đại 4×)", fontweight="bold")
ax3d.view_init(elev=18, azim=-60)
try:
    ax3d.set_box_aspect((2.6, 1.0, 1.7))
except AttributeError:
    pass  # ban matplotlib cu khong co set_box_aspect, bo qua khong loi

mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
fig3d.colorbar(mappable, ax=ax3d, shrink=0.6, pad=0.08, label="Áp lực gió W(z) [kPa]")

fig3d.tight_layout()
fig3d.savefig("fig_3d_buildings.png", dpi=200)

print("Da ve xong mo hinh 3D ca 3 toa nha -> fig_3d_buildings.png")
print("Chay file nay bang 'python simulate_3d_full.py' (may co man hinh) de xoay cua so 3D bang chuot.")

plt.show()  # mo tat ca cua so bieu do + mo hinh 3D, xoay duoc bang chuot
