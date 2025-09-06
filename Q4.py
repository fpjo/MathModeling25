import numpy as np
from scipy.optimize import brentq, differential_evolution
import warnings
import time
import pandas as pd
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 定义常量与初始条件 ---
P_M0 = np.array([20000.0, 0.0, 2000.0]); V_M_SCALAR = 300.0
# 无人机初始位置
DRONE_POSITIONS = {
    'FY1': np.array([17800.0, 0, 1800]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0])
}
R_CLOUD = 10.0; V_SINK = np.array([0.0, 0.0, -3.0]); T_EFFECT = 20.0
R_T = 7.0; H_T = 10.0
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0]);
G = np.array([0.0, 0.0, -9.8]);

# 预计算固定的导弹速度矢量
u_M = np.array([0.0, 0.0, 0.0]) - P_M0
v_m_vec = V_M_SCALAR * u_M / np.linalg.norm(u_M)

# --- 2. 核心计算函数 ---
def generate_target_points(num_angles=8):
    points = []; heights = [0, H_T / 2, H_T]; center_xy = P_T_BOTTOM_CENTER[:2]
    for h in heights:
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles); dx = R_T * np.cos(angle); dy = R_T * np.sin(angle)
            points.append(np.array([center_xy[0] + dx, center_xy[1] + dy, h]))
    return points
TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)


def calculate_combined_obscuration_time(strategies):
    """
    计算多个独立烟幕弹策略联合作用下的总遮蔽时间（并集）。
    'strategies' 是一个列表，每个元素是一个包含策略详情的字典。
    """
    bombs_info = []
    for strat in strategies:
        P_D0 = strat['drone_pos']
        theta, v_uav, t_drop, t_fuze = strat['params']
        
        v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
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
    if not all_event_times or len(all_event_times) < 2: return 0.0
    
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
    total_obscured_time = 0.0
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        if phi((t1 + t2) / 2.0) >= 0:
            total_obscured_time += (t2 - t1)
    return total_obscured_time

def get_obscuration_intervals(strategy):
    """
    计算并返回单个策略产生的遮蔽时间区间列表。
    """
    strat_list = [strategy]
    bombs_info = []
    for strat in strat_list:
        P_D0 = strat['drone_pos']
        theta, v_uav, t_drop, t_fuze = strat['params']
        v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
        p_drop = P_D0 + v_uav_vec * t_drop
        t_det = t_drop + t_fuze
        p_det = p_drop + v_uav_vec * t_fuze + 0.5 * G * t_fuze**2
        bombs_info.append({'p_det': p_det, 't_start_effect': t_det, 't_end_effect': t_det + T_EFFECT})
    
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
    if not all_event_times or len(all_event_times) < 2: return []

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
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        if phi((t1 + t2) / 2.0) >= 0:
            intervals.append((t1, t2))
    return intervals

# <<< FIX 1: Moved the objective function to the top level >>>
# It now accepts P_D0 as an extra argument to work with multiprocessing.
def objective_function_for_drone(params, p_d0_arg):
    theta, v_uav, t_drop, t_fuze = params
    t_det = t_drop + t_fuze
    missile_x_at_det = P_M0[0] + v_m_vec[0] * t_det
    drone_x_at_det = p_d0_arg[0] + v_uav * np.cos(theta) * t_det
    if missile_x_at_det <= drone_x_at_det + 10:
        return 1e9
    
    single_strat = [{'drone_pos': p_d0_arg, 'params': params}]
    return -calculate_combined_obscuration_time(single_strat)


# --- 3. 单无人机优化求解器 ---
def optimize_single_drone(drone_name, P_D0):
    """
    为单架无人机寻找最优投放策略（问题二求解器）。
    """
    print(f"\n--- 正在为无人机 {drone_name} 规划最优策略 ---")

    P_D0_XY = P_D0[:2]
    cos_for_bound_calc = -1.0 

    if drone_name == 'FY1':
        vec_to_target = P_T_BOTTOM_CENTER[:2] - P_D0_XY
        angle_to_target_rad = np.arctan2(vec_to_target[1], vec_to_target[0])
        lower_angle_bound_rad = angle_to_target_rad
        upper_angle_bound_rad = np.deg2rad(180)
        cos_for_bound_calc = np.cos(angle_to_target_rad) 
        
    elif drone_name == 'FY2':
        lower_angle_bound_rad = np.deg2rad(180)
        upper_angle_bound_rad = np.deg2rad(360)
        cos_for_bound_calc = np.cos(lower_angle_bound_rad) 
        
    elif drone_name == 'FY3':
        vec_to_origin = np.array([0.0, 0.0]) - P_D0_XY
        angle_to_origin_rad = np.arctan2(vec_to_origin[1], vec_to_origin[0])
        lower_angle_bound_rad = np.deg2rad(0)
        upper_angle_bound_rad = angle_to_origin_rad
        cos_for_bound_calc = np.cos(upper_angle_bound_rad) 
    
    print(f"为 {drone_name} 设定的角度搜索范围 (度): [{np.rad2deg(lower_angle_bound_rad):.2f}, {np.rad2deg(upper_angle_bound_rad):.2f}]")

    v_uav_for_bound = 140.0 * cos_for_bound_calc
    time_limit_conservative = (P_D0[0] - P_M0[0] + 10) / (v_m_vec[0] - v_uav_for_bound)
    if time_limit_conservative < 0 : time_limit_conservative = 80 
    
    t_drop_max = time_limit_conservative - 1
    t_fuze_max = time_limit_conservative
    if t_drop_max <= 0: t_drop_max = 15
    if t_fuze_max <= 1: t_fuze_max = 15

    bounds = [
        (lower_angle_bound_rad, upper_angle_bound_rad),
        (70, 140),
        (0, t_drop_max),
        (1, t_fuze_max)
    ]
    
    # <<< FIX 2: Call the global function and pass P_D0 via `args` >>>
    result = differential_evolution(
        objective_function_for_drone, 
        bounds, 
        args=(P_D0,),  # Pass P_D0 as an extra argument to the objective function
        strategy='best1bin', 
        maxiter=50, popsize=500,
        tol=0.01, mutation=(0.5, 1), recombination=0.8, 
        disp=True, workers=-1
    )
    
    return result.x, -result.fun

# --- 4. 主流程：解决问题四 ---
if __name__ == "__main__":
    print("="*60)
    print("           问题4：三无人机协同干扰策略求解           ")
    print("      策略：为各无人机应用定制剪枝策略 -> 分别寻优 -> 整合计算总效果      ")
    print("="*60)

    start_time_total = time.time()
    
    drones_to_use = ['FY1', 'FY2', 'FY3']
    optimal_strategies = []

    for name in drones_to_use:
        pos = DRONE_POSITIONS[name]
        best_params, max_time_individual = optimize_single_drone(name, pos)
        
        strategy = {
            'drone_name': name,
            'drone_pos': pos,
            'params': best_params,
            'individual_max_time': max_time_individual
        }
        optimal_strategies.append(strategy)
        print(f"--- {name} 最优策略规划完成, 可独立实现 {max_time_individual:.4f}s 遮蔽 ---")

    total_combined_time = calculate_combined_obscuration_time(optimal_strategies)
    
    end_time_total = time.time()
    
    # --- 5. 结果整理与输出 ---
    print("\n" + "="*60)
    print("               问题4 最终协同策略与结果               ")
    print("="*60)
    print(f"总计算耗时: {end_time_total - start_time_total:.2f} 秒")
    print(f"\n三架无人机协同部署，最终总有效遮蔽时间为: {total_combined_time:.4f} 秒")
    
    results_data = []
    for strat in optimal_strategies:
        name = strat['drone_name']
        theta_opt, v_uav_opt, t_drop_opt, t_fuze_opt = strat['params']
        p_d0 = strat['drone_pos']
        
        v_uav_vec_opt = np.array([v_uav_opt * np.cos(theta_opt), v_uav_opt * np.sin(theta_opt), 0])
        p_drop_opt = p_d0 + v_uav_vec_opt * t_drop_opt
        p_det_opt = p_drop_opt + v_uav_vec_opt * t_fuze_opt + 0.5 * G * t_fuze_opt**2
        
        results_data.append({
            '无人机编号': name,
            '飞行方向(度)': np.rad2deg(theta_opt),
            '飞行速度(m/s)': v_uav_opt,
            '投放时间(s)': t_drop_opt,
            '引信时间(s)': t_fuze_opt,
            '投放点X': p_drop_opt[0],
            '投放点Y': p_drop_opt[1],
            '投放点Z': p_drop_opt[2],
            '起爆点X': p_det_opt[0],
            '起爆点Y': p_det_opt[1],
            '起爆点Z': p_det_opt[2],
        })
        
    summary_df = pd.DataFrame(results_data)
    
    print("\n--- 各无人机最优策略详情 ---")
    print(summary_df.to_string(index=False, float_format="%.4f"))

    print("\n--- 各无人机独立遮蔽时间区间 ---")
    for strat in optimal_strategies:
        name = strat['drone_name']
        intervals = get_obscuration_intervals(strat)
        if not intervals:
            print(f"{name}: 未产生有效遮蔽")
        else:
            # 格式化输出，例如: "FY1: [ (31.42s, 35.88s) ]"
            interval_str = ", ".join([f"( {t1:.2f}s, {t2:.2f}s )" for t1, t2 in intervals])
            print(f"{name}: [ {interval_str} ]")

    print("\n" + "="*60)