import numpy as np
from scipy.optimize import brentq
import warnings
import time

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 定义常量与初始条件 ---
# 导弹M1初始位置与速度
P_M0 = np.array([20000.0, 0.0, 2000.0])
V_M_SCALAR = 300.0

# 无人机FY1初始位置 (根据您的结果，这是针对FY1的)
P_D0 = np.array([17800.0, 0, 1800])

# 烟幕云团属性
R_CLOUD = 10.0
V_SINK = np.array([0.0, 0.0, -3.0])
T_EFFECT = 20.0

# 保护目标属性 (圆柱体)
R_T = 7.0
H_T = 10.0
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0])

# 物理常量
G = np.array([0.0, 0.0, -9.8])

# 预计算固定的导弹速度矢量
u_M = np.array([0.0, 0.0, 0.0]) - P_M0
v_m_vec = V_M_SCALAR * u_M / np.linalg.norm(u_M)

# --- 2. 辅助函数 ---
def generate_target_points(num_angles=8):
    points = []; heights = [0, H_T / 2, H_T]; center_xy = P_T_BOTTOM_CENTER[:2]
    for h in heights:
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles); dx = R_T * np.cos(angle); dy = R_T * np.sin(angle)
            points.append(np.array([center_xy[0] + dx, center_xy[1] + dy, h]))
    return points

TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)

def get_obscuration_intervals(strategy_params):
    """
    根据给定的策略参数，计算并返回所有有效的遮蔽时间区间。
    """
    theta, v_uav, t_drop1, delta_t1, delta_t2, t_fuze1, t_fuze2, t_fuze3 = strategy_params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])

    t_drops = [t_drop1, t_drop1 + delta_t1, t_drop1 + delta_t1 + delta_t2]
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    
    bombs_info = []
    for i in range(3):
        t_drop = t_drops[i]
        t_fuze = fuze_times[i]
        p_drop = P_D0 + v_uav_vec * t_drop
        t_det = t_drop + t_fuze
        p_det = p_drop + v_uav_vec * t_fuze + 0.5 * G * t_fuze**2
        bombs_info.append({
            'p_det': p_det,
            't_start_effect': t_det,
            't_end_effect': t_det + T_EFFECT,
        })

    def get_missile_pos(t): return P_M0 + v_m_vec * t
    def get_cloud_pos(t, bomb_info): return bomb_info['p_det'] + V_SINK * (t - bomb_info['t_start_effect'])
    def phi(t):
        max_m_value = -1.0
        for bomb in bombs_info:
            if not (bomb['t_start_effect'] <= t <= bomb['t_end_effect']): continue
            p_m = get_missile_pos(t); p_cloud = get_cloud_pos(t, bomb); A = p_cloud - p_m; A_mag_sq = np.dot(A, A)
            if A_mag_sq < R_CLOUD**2: return 1.0
            A_mag = np.sqrt(A_mag_sq); cos_cone = np.sqrt(1 - R_CLOUD**2 / A_mag_sq)
            min_m_value_for_this_cloud = float('inf')
            for point in TARGET_CHECK_POINTS_24:
                vec = point - p_m; vec_mag = np.linalg.norm(vec)
                if vec_mag < 1e-9: cos_theta = 1.0
                else: cos_theta = np.dot(vec, A) / (vec_mag * A_mag)
                m = cos_theta - cos_cone
                if m < min_m_value_for_this_cloud: min_m_value_for_this_cloud = m
            if min_m_value_for_this_cloud > max_m_value: max_m_value = min_m_value_for_this_cloud
        return max_m_value

    all_event_times = set()
    for bomb in bombs_info:
        all_event_times.add(bomb['t_start_effect'])
        all_event_times.add(bomb['t_end_effect'])
    if not all_event_times or len(all_event_times) < 2: return [], 0.0

    t_scan_start, t_scan_end = min(all_event_times), max(all_event_times)
    roots = []
    scan_step = 0.1
    t_current = t_scan_start
    while t_current < t_scan_end:
        t_next = min(t_current + scan_step, t_scan_end)
        try:
            if phi(t_current) * phi(t_next) < 0:
                root = brentq(phi, t_current, t_next); roots.append(root)
        except (ValueError, RuntimeError): pass
        if t_next == t_scan_end: break
        t_current = t_next
    
    events = sorted(list(all_event_times.union(roots)))
    intervals = []
    total_time = 0.0
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        if phi((t1 + t2) / 2.0) >= 0:
            intervals.append((t1, t2))
            total_time += (t2 - t1)
    return intervals, total_time


if __name__ == "__main__":
    # --- 输入您提供的最优策略参数 ---
    theta_deg = 179.6541
    v_uav = 139.9541
    t_drop1 = 0.0123
    delta_t1 = 3.7466
    delta_t2 = 1.8907
    t_fuze1 = 3.6212
    t_fuze2 = 5.3636
    t_fuze3 = 6.0800

    # 将角度转换为弧度
    theta_rad = np.deg2rad(theta_deg)

    # 组合成策略参数列表
    final_strategy = [
        theta_rad, v_uav, t_drop1, delta_t1, delta_t2,
        t_fuze1, t_fuze2, t_fuze3
    ]

    print("="*60)
    print("      正在为给定的最优策略计算详细遮蔽时间段...")
    print("="*60)
    print("\n输入策略参数:")
    print(f"  飞行方向: {theta_deg:.4f} 度")
    print(f"  飞行速度: {v_uav:.4f} m/s")
    print(f"  首弹投放时间: {t_drop1:.4f} s")
    print(f"  投放间隔1: {delta_t1:.4f} s")
    print(f"  投放间隔2: {delta_t2:.4f} s")
    print(f"  引信时间 (1, 2, 3): {t_fuze1:.4f}s, {t_fuze2:.4f}s, {t_fuze3:.4f}s")
    
    # --- 执行计算并输出结果 ---
    start_time = time.time()
    obscuration_intervals, total_obscuration_time = get_obscuration_intervals(final_strategy)
    end_time = time.time()

    print(f"\n计算耗时: {end_time - start_time:.4f} 秒")
    print("-" * 60)
    print("\n计算结果:")
    if not obscuration_intervals:
        print("  该策略未产生有效遮蔽。")
    else:
        print("  产生的有效遮蔽时间段如下:")
        for i, (t1, t2) in enumerate(obscuration_intervals):
            duration = t2 - t1
            print(f"    - 区间 {i+1}: 从 {t1:.4f}s 到 {t2:.4f}s (持续 {duration:.4f}s)")
    
    print("-" * 60)
    print(f"\n验证：计算得到的总遮蔽时长为: {total_obscuration_time:.4f} 秒")
    print(f"      您提供的优化结果时长为: 7.6515 秒")
    print("="*60)