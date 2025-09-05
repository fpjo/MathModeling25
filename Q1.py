import numpy as np
from scipy.optimize import brentq
import warnings
import time

# --- MENTOR'S INSIGHT ---
# 模型版本: V3.0
# 本次升级的核心是将几何判据从8点检测升级为24点“金钟罩”模型。
# 1. 【顶部/底部】: 角度分辨率从90度提升至45度 (4点 -> 8点)。
# 2. 【中部】: 新增圆柱体“腰部”的8个检测点。
# 目标：以适度的计算成本增加，换取模型几何判据的极高鲁棒性，确保不漏判任何边缘遮蔽情况。

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 定义常量与初始条件 (源自题目描述) ---
P_M0 = np.array([20000.0, 0.0, 2000.0]); V_M_SCALAR = 300.0
P_D0 = np.array([17800.0, 0.0, 1800.0]); V_D_SCALAR = 120.0
T_DROP = 1.5; T_FUZE_DELAY = 3.6
R_CLOUD = 10.0; V_SINK = np.array([0.0, 0.0, -3.0]); T_EFFECT = 20.0
R_T = 7.0; H_T = 10.0
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0])
P_T_TOP_CENTER = np.array([0.0, 200.0, 10.0])
G = np.array([0.0, 0.0, -9.8])


# --- 2. 运动学预计算 ---
u_M = np.array([0.0, 0.0, 0.0]) - P_M0; V_M = V_M_SCALAR * u_M / np.linalg.norm(u_M)
u_D = np.array([0.0, 0.0, P_D0[2]]) - P_D0; V_D = V_D_SCALAR * u_D / np.linalg.norm(u_D) if np.linalg.norm(u_D) > 0 else np.array([0,0,0])
P_DROP = P_D0 + V_D * T_DROP; T_DET = T_DROP + T_FUZE_DELAY
V_GRENADE_AT_DROP = V_D; P_DET = P_DROP + V_GRENADE_AT_DROP * T_FUZE_DELAY + 0.5 * G * T_FUZE_DELAY**2
T_START = T_DET; T_END = T_DET + T_EFFECT


# --- 3. 定义动态函数 ---
def get_missile_pos(t):
    return P_M0 + V_M * t
def get_cloud_pos(t):
    if t < T_DET: return np.array([np.nan, np.nan, np.nan])
    return P_DET + V_SINK * (t - T_DET)

# --- 4. 核心：构建几何状态函数 Phi(t) ---
def generate_target_points(num_angles=8):
    """生成目标圆柱体的关键点集"""
    points = []
    heights = [0, H_T / 2, H_T] # 底部、中部、顶部
    center_xy = P_T_BOTTOM_CENTER[:2]

    for h in heights:
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles)
            dx = R_T * np.cos(angle)
            dy = R_T * np.sin(angle)
            points.append(np.array([center_xy[0] + dx, center_xy[1] + dy, h]))
            
    return points

# 预先生成一次即可，避免在phi函数中重复计算
TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)

def phi(t):
    """
    计算t时刻的遮蔽状态函数Phi(t)的值 (V3.0 24点高精度版)
    """
    # a. 获取当前时刻各物体位置
    p_m = get_missile_pos(t)
    p_cloud = get_cloud_pos(t)

    # b. 计算阴影锥的关键参数
    A = p_cloud - p_m
    A_mag_sq = np.dot(A, A)
    if A_mag_sq < R_CLOUD**2:
        return 1.0

    A_mag = np.sqrt(A_mag_sq)
    cos_cone = np.sqrt(1 - R_CLOUD**2 / A_mag_sq)

    # c. 检查所有24个关键点是否都在阴影锥内
    min_m_value = float('inf')

    for point in TARGET_CHECK_POINTS_24:
        vec = point - p_m
        vec_mag = np.linalg.norm(vec)
        
        if vec_mag < 1e-9:
            cos_theta = 1.0
        else:
            cos_theta = np.dot(vec, A) / (vec_mag * A_mag)
        
        m = cos_theta - cos_cone
        if m < min_m_value:
            min_m_value = m

    # d. 返回总状态函数值
    return min_m_value
# --- END OF MODIFICATION ---


# --- 5. 求解与计算 ---
def solve_problem_1():
    """执行求解过程：寻找根、判断区间、累计时长"""
    print("--- 问题1求解开始 ---")
    print(f"采用【修正后】的物理模型和【24点高精度】的几何判据。")
    print("-" * 30)
    print(f"干扰弹起爆时刻: {T_DET:.2f} s")
    print(f"烟幕有效时间窗口: [{T_START:.2f} s, {T_END:.2f} s]")
    print(f"起爆点坐标: ({P_DET[0]:.2f}, {P_DET[1]:.2f}, {P_DET[2]:.2f})")
    
    roots = []
    scan_step = 0.05
    t_current = T_START
    while t_current < T_END:
        t_next = min(t_current + scan_step, T_END)
        try:
            phi_current = phi(t_current)
            phi_next = phi(t_next)
            if phi_current * phi_next < 0:
                root = brentq(phi, t_current, t_next)
                roots.append(root)
        except (ValueError, RuntimeError): pass
        if t_next == T_END: break
        t_current = t_next
        
    events = sorted(list(set([T_START] + roots + [T_END])))
    print(f"\n找到 {len(roots)} 个状态转变时刻(根): {[f'{r:.3f}' for r in roots]}")

    total_obscured_time = 0.0
    print("\n分析各时间子区间:")
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        t_mid = (t1 + t2) / 2.0
        interval_length = t2 - t1
        if phi(t_mid) >= 0:
            status = "有效遮蔽"
            total_obscured_time += interval_length
        else:
            status = "未遮蔽"
        print(f" - 区间 [{t1:.3f}, {t2:.3f}], 时长 {interval_length:.3f}s, 状态: {status}")
    return total_obscured_time

# --- 6. 执行求解并输出结果 ---
if __name__ == "__main__":
    start_time = time.time()
    effective_time = solve_problem_1()
    end_time = time.time()
    
    print("\n" + "="*40)
    print(f"最终结果: 问题1的有效遮蔽总时长为: {effective_time:.8f} 秒")
    print(f"模型计算耗时: {end_time - start_time:.4f} 秒")
    print("="*40)
# 1.39198270