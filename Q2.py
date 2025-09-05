import numpy as np
from scipy.optimize import brentq, differential_evolution
import warnings
import time

# Differential Evolution

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- [常量定义 和 目标函数封装 部分与上一版完全相同，为简洁省略] ---
# --- 1. 定义常量与初始条件 (源自题目) ---
P_M0 = np.array([20000.0, 0.0, 2000.0]); V_M_SCALAR = 300.0
P_D0 = np.array([17800.0, 0.0, 1800.0]);
R_CLOUD = 10.0; V_SINK = np.array([0.0, 0.0, -3.0]); T_EFFECT = 20.0
R_T = 7.0; H_T = 10.0
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0]); P_T_TOP_CENTER = np.array([0.0, 200.0, 10.0]);
G = np.array([0.0, 0.0, -9.8]);
T_MAX = np.linalg.norm(P_M0)/V_M_SCALAR
# --- 2. 目标函数封装 ---
def generate_target_points(num_angles=8):
    points = []; heights = [0, H_T / 2, H_T]; center_xy = P_T_BOTTOM_CENTER[:2]
    for h in heights:
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles); dx = R_T * np.cos(angle); dy = R_T * np.sin(angle)
            points.append(np.array([center_xy[0] + dx, center_xy[1] + dy, h]))
    return points
TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)

def calculate_obscuration_time(params):
    theta, v_uav, t_drop, t_fuze = params; v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    p_drop = P_D0 + v_uav_vec * t_drop; t_det = t_drop + t_fuze
    p_det = p_drop + v_uav_vec * t_fuze + 0.5 * G * t_fuze**2
    t_start_effect = t_det; t_end_effect = t_det + T_EFFECT
    u_M = np.array([0.0, 0.0, 0.0]) - P_M0; v_m_vec = V_M_SCALAR * u_M / np.linalg.norm(u_M)
    def get_missile_pos(t): return P_M0 + v_m_vec * t
    def get_cloud_pos(t): return p_det + V_SINK * (t - t_det)
    def phi(t):
        if not (t_start_effect <= t <= t_end_effect): return -1.0
        p_m = get_missile_pos(t); p_cloud = get_cloud_pos(t); A = p_cloud - p_m; A_mag_sq = np.dot(A, A)
        if A_mag_sq < R_CLOUD**2: return 1.0
        A_mag = np.sqrt(A_mag_sq); cos_cone = np.sqrt(1 - R_CLOUD**2 / A_mag_sq)
        min_m_value = float('inf')
        for point in TARGET_CHECK_POINTS_24:
            vec = point - p_m; vec_mag = np.linalg.norm(vec)
            if vec_mag < 1e-9: cos_theta = 1.0
            else: cos_theta = np.dot(vec, A) / (vec_mag * A_mag)
            m = cos_theta - cos_cone
            if m < min_m_value: min_m_value = m
        return min_m_value
    roots = []; scan_step = 0.1; t_current = t_start_effect
    while t_current < t_end_effect:
        t_next = min(t_current + scan_step, t_end_effect)
        try:
            if phi(t_current) * phi(t_next) < 0:
                root = brentq(phi, t_current, t_next); roots.append(root)
        except (ValueError, RuntimeError): pass
        if t_next == t_end_effect: break
        t_current = t_next
    events = sorted(list(set([t_start_effect] + roots + [t_end_effect])))
    total_obscured_time = 0.0
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        if phi((t1 + t2) / 2.0) >= 0: total_obscured_time += (t2 - t1)
    return total_obscured_time

def objective_function(params):
    return -calculate_obscuration_time(params)

# --- 3. 执行优化求解 ---
if __name__ == "__main__":
    print("--- 问题2求解开始 (模型版本: V3.2 - 约束搜索空间) ---")
    print("正在使用差分进化算法在预估的“关键窗口”内寻找最优策略...")
    
    # --- MENTOR'S MODIFICATION ---
    # 根据物理直觉，为决策变量设定更精确的搜索边界
    bounds = [
        (np.deg2rad(0), np.deg2rad(180)), # 飞行方向: 主要朝向目标所在一侧 [0, 180]度
        (70, 140),                      # 飞行速度: 倾向于高速以尽快到达拦截位置
        (0, 65),                        # 投放时间: 确保能在导弹接近时投放
        (1, 10)                          # 引信时间: 确保合理的下落和起爆时间
    ]
    # --- END OF MODIFICATION ---

    start_time = time.time()
    
    # 增加迭代次数和种群大小，进行更充分的搜索
    result = differential_evolution(
        objective_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=100, 
        popsize=3000, 
        tol=0.01, 
        mutation=(0.5, 1), 
        recombination=0.8, 
        disp=True,
        workers=16
    )
    
    end_time = time.time()

    # --- 4. 结果展示 ---
    best_params = result.x; max_time = -result.fun
    print("\n" + "="*50); print("          问题2：最优烟幕投放策略          "); print("="*50)
    print(f"优化过程耗时: {end_time - start_time:.2f} 秒")
    print(f"\n找到的最优策略可获得最长有效遮蔽时间: {max_time:.4f} 秒")
    print("\n--- 最优决策变量 ---")
    print(f"无人机飞行方向 (角度): {np.rad2deg(best_params[0]):.2f} 度")
    print(f"无人机飞行速度: {best_params[1]:.2f} m/s")
    print(f"烟幕弹投放时间: {best_params[2]:.2f} s")
    print(f"烟幕弹引信时间: {best_params[3]:.2f} s")
    theta_opt, v_uav_opt, t_drop_opt, t_fuze_opt = best_params
    v_uav_vec_opt = np.array([v_uav_opt * np.cos(theta_opt), v_uav_opt * np.sin(theta_opt), 0])
    p_drop_opt = P_D0 + v_uav_vec_opt * t_drop_opt
    p_det_opt = p_drop_opt + v_uav_vec_opt * t_fuze_opt + 0.5 * G * t_fuze_opt**2
    print("\n--- 对应的策略结果 ---")
    print(f"烟幕弹投放点坐标: ({p_drop_opt[0]:.2f}, {p_drop_opt[1]:.2f}, {p_drop_opt[2]:.2f})")
    print(f"烟幕弹起爆点坐标: ({p_det_opt[0]:.2f}, {p_det_opt[1]:.2f}, {p_det_opt[2]:.2f})")
    print("="*50)
# f(x)= -4.530591853241796
# 网络最优 4.6 4.8
# 进化过程耗时: 2185.40 秒

# 找到的最优策略可获得最长有效遮蔽时间: 4.5344 秒

# --- 最优决策变量 ---
# 无人机飞行方向 (角度): 177.54 度
# 无人机飞行速度: 79.60 m/s
# 烟幕弹投放时间: 0.14 s
# 烟幕弹引信时间: 2.70 s

# --- 对应的策略结果 ---
# 烟幕弹投放点坐标: (17788.49, 0.49, 1800.00)
# 烟幕弹起爆点坐标: (17573.60, 9.72, 1764.22)