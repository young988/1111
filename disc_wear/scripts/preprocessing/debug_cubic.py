import numpy as np

# 测试参数
delta_r_original = 0.1
T = 25.4
R = 241.3
theta_deg = 20
N = 1

theta_rad = np.radians(theta_deg)
tan_theta = np.tan(theta_rad)

# 正向计算体积
term1 = T * R
term2 = -(T / 2) * delta_r_original
term3 = R * tan_theta * delta_r_original
term4 = -(2 / 3) * tan_theta * (delta_r_original ** 2)
volume = 2 * np.pi * delta_r_original * (term1 + term2 + term3 + term4) * N

print(f"原始半径: {delta_r_original} mm")
print(f"计算体积: {volume} mm³")
print()

# 反向求解
a = -(2/3) * tan_theta
b = R * tan_theta - T/2
c = T * R
d = -volume / (2 * np.pi * N)

print(f"三次方程系数:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  c = {c}")
print(f"  d = {d}")
print()

# 求解
coefficients = [a, b, c, d]
roots = np.roots(coefficients)

print(f"所有根:")
for i, root in enumerate(roots):
    print(f"  根{i+1}: {root}")
    print(f"    实部: {np.real(root)}")
    print(f"    虚部: {np.imag(root)}")
    print(f"    是否为实数: {np.abs(np.imag(root)) < 1e-10}")
    print(f"    是否为正数: {np.real(root) > 0}")
print()

# 验证每个根
print("验证每个根:")
for i, root in enumerate(roots):
    r = np.real(root)
    if np.abs(np.imag(root)) < 1e-10 and r > 0:
        # 重新计算体积
        term1_check = T * R
        term2_check = -(T / 2) * r
        term3_check = R * tan_theta * r
        term4_check = -(2 / 3) * tan_theta * (r ** 2)
        volume_check = 2 * np.pi * r * (term1_check + term2_check + term3_check + term4_check) * N
        print(f"  根{i+1} (r={r:.6f}): 体积={volume_check:.6f}, 误差={abs(volume_check - volume):.6e}")
