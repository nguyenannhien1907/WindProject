import numpy as np

num_floors = 30
floor_height = 3.5
b = 30.0

floors = np.arange(1, num_floors + 1)
z = floors * floor_height

W0 = 1.25
alpha_terrain = 0.16
zg = 300.0
k_z = 2.01 * (np.maximum(z, 3.0) / zg) ** (2 * alpha_terrain)

c_square, c_chamfer, c_round = 1.4, 1.1, 0.95
p_square = W0 * k_z * c_square
p_chamfer = W0 * k_z * c_chamfer
p_round = W0 * k_z * c_round

A_floor = b * floor_height  # diện tích đón gió 1 tầng (m²)
F_square = p_square * A_floor
F_chamfer = p_chamfer * A_floor
F_round = p_round * A_floor

k_story = 3.5e5  # độ cứng mỗi tầng (kN/m)
K = np.zeros((num_floors, num_floors))  # ma trận 30×30, toàn số 0

for i in range(num_floors):
    if i == num_floors - 1:  # tầng trên cùng: chỉ 1 lò xo
        K[i, i] = k_story
    else:  # các tầng khác: 2 lò xo
        K[i, i] = 2 * k_story
        K[i, i + 1] = -k_story
        K[i + 1, i] = -k_story

u_square = np.linalg.solve(K, F_square) * 1000  # đổi m → mm
u_chamfer = np.linalg.solve(K, F_chamfer) * 1000
u_round = np.linalg.solve(K, F_round) * 1000

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(u_square, z, '-o', label='Vuông')
ax.plot(u_chamfer, z, '-s', label='Vát góc')
ax.plot(u_round, z, '-^', label='Bo tròn')
ax.set_xlabel('Chuyển vị ngang (mm)')
ax.set_ylabel('Độ cao (m)')
ax.legend()

plt.savefig('fig_displacement.png', dpi=200)
plt.show()  # mở cửa sổ hiện biểu đồ