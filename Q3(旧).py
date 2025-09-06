import numpy as np
from scipy.optimize import brentq, differential_evolution
import warnings
import time
import pandas as pd
import os

# 忽略计算中可能出现的无效值警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 定义常量与初始条件 (源自题目) ---
# 导弹M1初始位置与速度
P_M0 = np.array([20000.0, 0.0, 2000.0])
V_M_SCALAR = 300.0

# 无人机FY1初始位置
P_D0 = np.array([17800.0, 0.0, 1800.0])

# 烟幕云团属性
R_CLOUD = 10.0  # 烟幕云团有效半径
V_SINK = np.array([0.0, 0.0, -3.0])  # 烟幕云团下沉速度
T_EFFECT = 20.0  # 单个云团有效遮蔽持续时间

# 保护目标属性 (圆柱体)
R_T = 7.0  # 半径
H_T = 10.0  # 高度
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0]) # 底面圆心

# 物理常量
G = np.array([0.0, 0.0, -9.8]) # 重力加速度

# <<< FIX: Pre-calculate missile velocity vector globally >>>
# The missile's path is fixed, so its velocity vector is a constant.
u_M = np.array([0.0, 0.0, 0.0]) - P_M0
v_m_vec = V_M_SCALAR * u_M / np.linalg.norm(u_M)

# --- 2. 目标函数封装 ---

def generate_target_points(num_angles=8):
    """在圆柱体目标表面生成离散的检查点，用于判断是否被遮蔽"""
    points = []
    heights = [0, H_T / 2, H_T] # 在底部、中部、顶部高度检查
    center_xy = P_T_BOTTOM_CENTER[:2]
    for h in heights:
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles)
            dx = R_T * np.cos(angle)
            dy = R_T * np.sin(angle)
            points.append(np.array([center_xy[0] + dx, center_xy[1] + dy, h]))
    return points

TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)

def calculate_total_obscuration_time_multi(params):
    """
    计算三枚烟幕弹联合作用下的总有效遮蔽时间。
    (此函数内部逻辑无需修改)
    """
    theta, v_uav, t_drop1, t_fuze1, t_fuze2, t_fuze3 = params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    bombs_info = []
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    for i in range(3):
        t_drop = t_drop1 + i
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
        p_m = get_missile_pos(t); max_m_value = -1.0
        for bomb in bombs_info:
            if not (bomb['t_start_effect'] <= t <= bomb['t_end_effect']): continue
            p_cloud = get_cloud_pos(t, bomb); A = p_cloud - p_m; A_mag_sq = np.dot(A, A)
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
    for bomb in bombs_info: all_event_times.add(bomb['t_start_effect']); all_event_times.add(bomb['t_end_effect'])
    if not all_event_times: return 0.0
    t_scan_start, t_scan_end = min(all_event_times), max(all_event_times); roots = []
    scan_step = 0.1; t_current = t_scan_start
    while t_current < t_scan_end:
        t_next = min(t_current + scan_step, t_scan_end)
        try:
            if phi(t_current) * phi(t_next) < 0: root = brentq(phi, t_current, t_next); roots.append(root)
        except (ValueError, RuntimeError): pass
        if t_next == t_scan_end: break
        t_current = t_next
    events = sorted(list(all_event_times.union(roots))); total_obscured_time = 0.0
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        if phi((t1 + t2) / 2.0) >= 0: total_obscured_time += (t2 - t1)
    return total_obscured_time

# <<< MODIFICATION 1: Re-introducing the penalty function >>>
def objective_function_multi(params):
    """
    优化器调用的目标函数。
    在计算前检查动态物理约束，如果违反则返回巨大惩罚值。
    """
    theta, v_uav, t_drop1, t_fuze1, t_fuze2, t_fuze3 = params
    
    # 动态约束检查
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    for i in range(3):
        t_drop = t_drop1 + i
        t_det = t_drop + fuze_times[i]
        
        # 检查引爆时间点是否满足物理约束
        missile_x_at_det = P_M0[0] + v_m_vec[0] * t_det
        drone_x_at_det = P_D0[0] + v_uav_vec[0] * t_det # 此处用无人机位置近似烟幕弹位置
        if missile_x_at_det <= drone_x_at_det + 10: # 如果导弹已追上或超过烟幕弹
            return 1e9  # 返回巨大惩罚值
            
    # 如果所有约束都满足，则正常计算目标函数值
    return -calculate_total_obscuration_time_multi(params)

# --- 3. 执行优化求解 ---
if __name__ == "__main__":
    print("--- 问题3求解开始 (保守Bounds + 动态约束惩罚) ---")
    
    # --- 角度搜索范围设定 ---
    P_D0_XY = P_D0[:2]; P_T_CENTER_XY = P_T_BOTTOM_CENTER[:2]
    vec_to_target = P_T_CENTER_XY - P_D0_XY
    angle_to_target_rad = np.arctan2(vec_to_target[1], vec_to_target[0])
    lower_angle_bound_rad = angle_to_target_rad
    upper_angle_bound_rad = np.deg2rad(180)

    # <<< MODIFICATION 2: Corrected logic for conservative bounds >>>
    # --- 计算并设定保守的时间Bounds ---
    # 为了确保不错过任何解，硬边界应由最晚可能的失效时间决定。
    # 这发生在无人机以最快速度(140m/s)飞行时。
    v_uav_for_bound = 140.0 * np.cos(upper_angle_bound_rad)
    time_limit_conservative = (P_D0[0] - P_M0[0] + 10) / (v_m_vec[0] - v_uav_for_bound)

    # 根据最晚失效时间，设定一个安全的、宽容的搜索边界
    t_drop1_max = time_limit_conservative - 2
    t_fuze_max = time_limit_conservative # 引信时间本身可以长一些，只要保证引爆点有效
    
    # 增加一个检查，确保计算出的边界为正
    if t_drop1_max <= 0:
        t_drop1_max = 10 # 如果计算出错，给一个默认值
        print("警告: 计算出的投放时间上限为负，已重置为默认值。")
    if t_fuze_max <= 1:
        t_fuze_max = 10 # 引信时间下限为1
        print("警告: 计算出的引信时间上限过小，已重置为默认值。")


    print("\n--- 优化参数设定 ---")
    print(f"飞行角度搜索范围: [{np.rad2deg(lower_angle_bound_rad):.2f}, {np.rad2deg(upper_angle_bound_rad):.2f}] 度")
    print(f"安全的拦截时机上限 (基于140m/s计算): {time_limit_conservative:.2f} 秒")
    print(f"投放时间 t_drop1 上限设为: {t_drop1_max:.2f} 秒")
    print(f"引信时间 t_fuze 上限设为: {t_fuze_max:.2f} 秒")

    # 为6个决策变量设定更紧的搜索边界
    bounds = [
        (lower_angle_bound_rad, upper_angle_bound_rad),
        (70, 140),
        (0, t_drop1_max),
        (1, t_fuze_max),
        (1, t_fuze_max),
        (1, t_fuze_max)
    ]

    start_time = time.time()
    
    # 运行差分进化算法
    result = differential_evolution(
        objective_function_multi, 
        bounds, 
        strategy='best1bin', 
        maxiter=100,      # 迭代次数
        popsize=3000,       # 种群大小
        tol=0.01, 
        mutation=(0.5, 1), 
        recombination=0.8, 
        disp=True,         # 显示优化过程
        workers=-1         # 使用所有可用的CPU核心
    )
    
    end_time = time.time()

    # --- 4. 结果整理与输出 ---
    best_params = result.x
    max_time = -result.fun
    
    # 如果最优解依然是惩罚值，说明没有找到可行解
    if max_time < 0:
         print("\n" + "="*60)
         print("           警告：未能找到满足所有约束的可行解           ")
         print("可能是约束过于严格或搜索不充分。请尝试增加优化器迭代次数/种群大小。")
         print("="*60)
    else:
        print("\n" + "="*60)
        print("           问题3：最优烟幕投放策略 (3枚干扰弹)           ")
        print("="*60)
        print(f"优化过程耗时: {end_time - start_time:.2f} 秒")
        print(f"\n找到的最优策略可获得最长有效遮蔽时间: {max_time:.4f} 秒")
        
        # 解析最优参数
        theta_opt, v_uav_opt, t_drop1_opt, t_fuze1_opt, t_fuze2_opt, t_fuze3_opt = best_params
        v_uav_vec_opt = np.array([v_uav_opt * np.cos(theta_opt), v_uav_opt * np.sin(theta_opt), 0])

        print("\n--- 无人机飞行策略 ---")
        print(f"飞行方向 (角度): {np.rad2deg(theta_opt):.2f} 度")
        print(f"飞行速度: {v_uav_opt:.2f} m/s")

        # 准备数据
        results_data = []
        bombs_info = []
        fuze_times_opt = [t_fuze1_opt, t_fuze2_opt, t_fuze3_opt]
        
        for i in range(3):
            t_drop_opt = t_drop1_opt + i
            t_fuze_opt = fuze_times_opt[i]
            t_det = t_drop_opt + t_fuze_opt
            bombs_info.append({
                't_start_effect': t_det,
                't_end_effect': t_det + T_EFFECT
            })
        print("\n--- 各干扰弹遮蔽时间段 ---")
        for i, b in enumerate(bombs_info):
            print(f"干扰弹 {i+1}: {b['t_start_effect']:.2f} s  到  {b['t_end_effect']:.2f} s")

        print("\n--- 各干扰弹投放详情 ---")
        for i in range(3):
            t_drop_opt = t_drop1_opt + i
            t_fuze_opt = fuze_times_opt[i]

            p_drop_opt = P_D0 + v_uav_vec_opt * t_drop_opt
            p_det_opt = p_drop_opt + v_uav_vec_opt * t_fuze_opt + 0.5 * G * t_fuze_opt**2
            
            results_data.append({
                '干扰弹编号': f'弹_{i+1}',
                '投放时间(s)': t_drop_opt,
                '引信时间(s)': t_fuze_opt,
                '投放点X': p_drop_opt[0],
                '投放点Y': p_drop_opt[1],
                '投放点Z': p_drop_opt[2],
                '起爆点X': p_det_opt[0],
                '起爆点Y': p_det_opt[1],
                '起爆点Z': p_det_opt[2],
            })

        # 创建DataFrame
        df = pd.DataFrame(results_data)
        summary_df = pd.DataFrame({
            '无人机飞行方向(度)': [np.rad2deg(theta_opt)],
            '无人机飞行速度(m/s)': [v_uav_opt],
            '最长总遮蔽时间(s)': [max_time]
        })
        
        # 输出到终端
        print("\n" + "="*60)
        print("概要与结果")
        print(summary_df.to_string(index=False, float_format="%.4f"))
        print("\n各干扰弹投放详情")
        print(df.to_string(index=False, float_format="%.4f"))
        print("="*60)