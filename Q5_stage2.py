import numpy as np
from scipy.optimize import differential_evolution, brentq
import warnings
import time
import pandas as pd

# --- 0. 环境设置 ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# --- 1. 定义常量与初始条件 (源自题目) ---
R_T = 7.0
H_T = 10.0
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0])
R_CLOUD = 10.0
V_SINK = np.array([0.0, 0.0, -3.0])
T_EFFECT = 20.0
G = np.array([0.0, 0.0, -9.8])

MISSILES_INFO = {
    'M1': {'P0': np.array([20000.0, 0.0, 2000.0])},
    'M2': {'P0': np.array([19000.0, 600.0, 2100.0])},
    'M3': {'P0': np.array([18000.0, -600.0, 1900.0])}
}

DRONES_INFO = {
    'FY1': {'P0': np.array([17800.0, 0.0, 1800.0])},
    'FY2': {'P0': np.array([12000.0, 1400.0, 1400.0])},
    'FY3': {'P0': np.array([6000.0, -3000.0, 700.0])},
    'FY4': {'P0': np.array([11000.0, 2000.0, 1800.0])},
    'FY5': {'P0': np.array([13000.0, -2000.0, 1300.0])}
}

V_M_SCALAR = 300.0
FAKE_TARGET_POS = np.array([0.0, 0.0, 0.0])
for m_id, m_data in MISSILES_INFO.items():
    u_M = FAKE_TARGET_POS - m_data['P0']
    m_data['V_VEC'] = V_M_SCALAR * u_M / np.linalg.norm(u_M)

# --- 1.1. 定义基于参考方案的预设策略（作为局部搜索的中心点）---
PRESET_STRATEGIES = {
    ('FY1', 'M1'): {'theta_deg': 179.65, 'v_uav': 139.95, 'theta_rad': np.deg2rad(179.65)},
    ('FY3', 'M1'): {'theta_deg': 74.30,  'v_uav': 136.38, 'theta_rad': np.deg2rad(74.30)},
    ('FY4', 'M1'): {'theta_deg': -57.67, 'v_uav': 131.86, 'theta_rad': np.deg2rad(-57.67)},
    ('FY2', 'M2'): {'theta_deg': -66.88, 'v_uav': 137.97, 'theta_rad': np.deg2rad(-66.88)},
    ('FY5', 'M3'): {'theta_deg': 116.91, 'v_uav': 139.07, 'theta_rad': np.deg2rad(116.91)},
}
FIXED_ASSIGNMENT = {
    'M1': ['FY3', 'FY4'],
    'M2': ['FY2'],
    'M3': ['FY5']
}

# --- 2. 核心计算与优化函数 ---

def generate_target_points(num_angles=8):
    points = []
    heights = [0, H_T / 2, H_T]
    center_xy = P_T_BOTTOM_CENTER[:2]
    for h in heights:
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles)
            dx = R_T * np.cos(angle)
            dy = R_T * np.sin(angle)
            points.append(np.array([center_xy[0] + dx, center_xy[1] + dy, h]))
    return points

TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)

def calculate_obscuration_details(params, P_D0, P_M0, v_m_vec, bomb_indices=None):
    theta, v_uav, t_drop1, delta_t1, delta_t2, t_fuze1, t_fuze2, t_fuze3 = params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])

    t_drops = [t_drop1, t_drop1 + delta_t1, t_drop1 + delta_t1 + delta_t2]
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    
    all_bombs_info = []
    for i in range(3):
        t_drop = t_drops[i]
        t_fuze = fuze_times[i]
        t_det = t_drop + t_fuze
        p_drop = P_D0 + v_uav_vec * t_drop
        p_det = p_drop + v_uav_vec * t_fuze + 0.5 * G * t_fuze**2
        
        all_bombs_info.append({
            'p_drop': p_drop, 'p_det': p_det,
            't_start_effect': t_det, 't_end_effect': t_det + T_EFFECT,
        })

    if bomb_indices is not None:
        bombs_to_process = [all_bombs_info[i] for i in bomb_indices if i < len(all_bombs_info)]
    else:
        bombs_to_process = all_bombs_info

    def get_missile_pos(t): return P_M0 + v_m_vec * t
    def get_cloud_pos(t, bomb_info): return bomb_info['p_det'] + V_SINK * (t - bomb_info['t_start_effect'])
    
    def phi(t):
        p_m = get_missile_pos(t)
        max_m_value = -1.0
        is_obscured = False
        for bomb in bombs_to_process:
            if not (bomb['t_start_effect'] <= t <= bomb['t_end_effect']): continue
            p_cloud = get_cloud_pos(t, bomb)
            A = p_cloud - p_m
            A_mag_sq = np.dot(A, A)
            if A_mag_sq < R_CLOUD**2:
                is_obscured = True
                break
            A_mag = np.sqrt(A_mag_sq)
            cos_cone = np.sqrt(1 - R_CLOUD**2 / A_mag_sq)
            min_m_value_for_this_cloud = float('inf')
            for point in TARGET_CHECK_POINTS_24:
                vec = point - p_m
                vec_mag = np.linalg.norm(vec)
                if vec_mag < 1e-9: cos_theta = 1.0
                else: cos_theta = np.dot(vec, A) / (vec_mag * A_mag)
                m = cos_theta - cos_cone
                min_m_value_for_this_cloud = min(min_m_value_for_this_cloud, m)
            max_m_value = max(max_m_value, min_m_value_for_this_cloud)
        
        if is_obscured: return 1.0
        return max_m_value

    all_event_times = set()
    for bomb in bombs_to_process:
        all_event_times.add(bomb['t_start_effect'])
        all_event_times.add(bomb['t_end_effect'])
    
    if not all_event_times: return 0.0, []
    
    t_scan_start, t_scan_end = min(all_event_times), max(all_event_times)
    roots = []
    scan_step = 0.1
    t_current = t_scan_start
    while t_current < t_scan_end:
        t_next = min(t_current + scan_step, t_scan_end)
        try:
            if phi(t_current) * phi(t_next) < 0:
                root = brentq(phi, t_current, t_next)
                roots.append(root)
        except (ValueError, RuntimeError): pass
        if t_next == t_scan_end: break
        t_current = t_next
        
    events = sorted(list(all_event_times.union(roots)))
    total_obscured_time = 0.0
    obscured_intervals = []
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        t_mid = (t1 + t2) / 2.0
        if phi(t_mid) >= 0:
            duration = t2 - t1
            total_obscured_time += duration
            if obscured_intervals and abs(obscured_intervals[-1][1] - t1) < 1e-6:
                obscured_intervals[-1][1] = t2
            else:
                obscured_intervals.append([t1, t2])
                
    return total_obscured_time, obscured_intervals

def objective_function(params, P_D0, P_M0, v_m_vec):
    theta, v_uav, t_drop1, delta_t1, delta_t2, t_fuze1, t_fuze2, t_fuze3 = params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    
    t_drops = [t_drop1, t_drop1 + delta_t1, t_drop1 + delta_t1 + delta_t2]
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    
    for i in range(3):
        t_drop = t_drops[i]
        t_det = t_drop + fuze_times[i]
        
        missile_x_at_det = P_M0[0] + v_m_vec[0] * t_det
        drone_x_at_drop = P_D0[0] + v_uav_vec[0] * t_drop
        
        if missile_x_at_det <= drone_x_at_drop:
            return 1e9

    total_time, _ = calculate_obscuration_details(params, P_D0, P_M0, v_m_vec)
    return -total_time

def optimize_strategy_locally(drone_id, missile_id):
    drone = DRONES_INFO[drone_id]
    missile = MISSILES_INFO[missile_id]
    P_D0, P_M0, v_m_vec = drone['P0'], missile['P0'], missile['V_VEC']
    
    preset = PRESET_STRATEGIES.get((drone_id, missile_id))
    if not preset:
        print(f"错误：在PRESET_STRATEGIES中未找到组合 [{drone_id} -> {missile_id}] 的策略。")
        return None
    
    theta_center_rad = preset['theta_rad']
    v_uav_center = preset['v_uav']

    print(f"\n--- 正在为组合 [无人机 {drone_id} -> 导弹 {missile_id}] 进行局部优化... ---")
    print(f"    搜索中心点: 方向 = {preset['theta_deg']:.2f}度, 速度 = {v_uav_center:.2f} m/s")

    # --- 定义局部搜索范围 ---
    THETA_RANGE_DEG = 3.0
    V_UAV_RANGE = 5.0
    
    theta_bound = (theta_center_rad - np.deg2rad(THETA_RANGE_DEG), theta_center_rad + np.deg2rad(THETA_RANGE_DEG))
    v_uav_bound = (max(70.0, v_uav_center - V_UAV_RANGE), min(140.0, v_uav_center + V_UAV_RANGE))

    # --- MODIFICATION: 仿照Q3.py，计算更精细的动态时间边界 ---
    # 确定局部搜索范围内，无人机在x轴上的“最快”逆行速度分量
    cos_theta_worst = min(np.cos(theta_bound[0]), np.cos(theta_bound[1]))
    v_uav_for_bound_calc = v_uav_bound[1] * cos_theta_worst # 使用速度上限和最逆行的角度
    
    # 避免分母为零或正数的情况
    if (v_m_vec[0] - v_uav_for_bound_calc) >= 0:
        time_limit_conservative = 100.0 # 若无人机无法在x轴上快过导弹，给一个较宽松的默认值
    else:
        time_limit_conservative = (P_D0[0] - P_M0[0]) / (v_m_vec[0] - v_uav_for_bound_calc)

    t_drop1_max = max(1.0, time_limit_conservative - 2.0) # 减去最小的两个间隔
    t_fuze_max = max(1.0, time_limit_conservative)
    print(f"    预估拦截时机上限: {time_limit_conservative:.2f}s, 投放/引信时间将在此范围内搜索。")
    
    bounds = [
        theta_bound,
        v_uav_bound,
        (0.1, t_drop1_max),
        (1.0, 5.0),
        (1.0, 5.0),
        (1.0, t_fuze_max),
        (1.0, t_fuze_max),
        (1.0, t_fuze_max)
    ]
    
    # --- MODIFICATION: 仿照Q3.py，采用大规模种群进行优化 ---
    result = differential_evolution(
        objective_function, bounds, 
        args=(P_D0, P_M0, v_m_vec),
        strategy='best1bin', 
        maxiter=100,      # Q3.py: 100
        popsize=100,     # Q3.py: 3000
        tol=0.01, 
        mutation=(0.5, 1), 
        recombination=0.8,# Q3.py: 0.8
        disp=True,        # Q3.py: True
        workers=-1
    )
    
    max_time = -result.fun
    best_params = result.x
    
    print(f"组合 [{drone_id} -> {missile_id}] 优化完成。最长遮蔽时间: {max_time:.4f} 秒")
    
    if max_time <= 0:
        return {
            'drone_id': drone_id, 'missile_id': missile_id,
            'max_time': 0, 'params': None, 'intervals': [],
            'individual_bomb_details': []
        }

    _, intervals = calculate_obscuration_details(best_params, P_D0, P_M0, v_m_vec)
    
    individual_bomb_details = []
    for i in range(3):
        duration, bomb_intervals = calculate_obscuration_details(best_params, P_D0, P_M0, v_m_vec, bomb_indices=[i])
        individual_bomb_details.append({
            'duration': duration,
            'intervals': bomb_intervals
        })

    return {
        'drone_id': drone_id, 'missile_id': missile_id,
        'max_time': max_time, 'params': best_params, 'intervals': intervals,
        'individual_bomb_details': individual_bomb_details
    }

def calculate_union_of_intervals(intervals_list):
    if not intervals_list: return 0, []
    flat_list = [interval for sublist in intervals_list for interval in sublist]
    if not flat_list: return 0, []
    flat_list.sort(key=lambda x: x[0])
    
    merged = []
    if flat_list:
        current_start, current_end = flat_list[0]
        for next_start, next_end in flat_list[1:]:
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged.append([current_start, current_end])
                current_start, current_end = next_start, next_end
        merged.append([current_start, current_end])
    
    total_duration = sum(end - start for start, end in merged)
    return total_duration, merged

# --- 3. 主执行流程 ---
if __name__ == "__main__":
    start_time_total = time.time()
    
    print("="*65 + "\n      问题5：基于参考方案的局部精细化搜索 (大规模种群)      \n" + "="*65)
    print("说明: 采用Q3优化思路，使用大规模种群和动态时间边界进行微调。")

    optimized_results = {}
    
    for missile_id, assigned_drones in FIXED_ASSIGNMENT.items():
        for drone_id in assigned_drones:
            if (drone_id, missile_id) in PRESET_STRATEGIES:
                result = optimize_strategy_locally(drone_id, missile_id)
                optimized_results[(drone_id, missile_id)] = result
            else:
                 print(f"\n警告：组合 [{drone_id} -> {missile_id}] 不在 PRESET_STRATEGIES 中，已跳过。")

    print("\n" + "="*70 + "\n            问题5：最终优化后的烟幕投放策略与结果            \n" + "="*70)
    
    if not optimized_results:
        print("错误：未能生成任何有效的优化结果。")
    else:
        missile_final_coverage = {}
        total_obscuration_time_all_missiles = 0
        
        for missile_id, assigned_drones in FIXED_ASSIGNMENT.items():
            intervals_for_missile = []
            for drone_id in assigned_drones:
                result = optimized_results.get((drone_id, missile_id))
                if result and result['intervals']:
                    intervals_for_missile.append(result['intervals'])
            
            union_time, merged_intervals = calculate_union_of_intervals(intervals_for_missile)
            missile_final_coverage[missile_id] = {
                'total_time': union_time,
                'merged_intervals': merged_intervals
            }
            total_obscuration_time_all_missiles += union_time

        print(f"\n最优分配方案下的最大总遮蔽时间: {total_obscuration_time_all_missiles:.4f} 秒\n")
        final_output_data = []

        for missile_id, assigned_drones in FIXED_ASSIGNMENT.items():
            if not assigned_drones: continue

            missile_coverage_info = missile_final_coverage.get(missile_id, {})
            total_missile_time = missile_coverage_info.get('total_time', 0)
            merged_intervals = missile_coverage_info.get('merged_intervals', [])
            intervals_str = ', '.join([f"[{start:.2f}s, {end:.2f}s]" for start, end in merged_intervals])
            
            for drone_id in assigned_drones:
                result = optimized_results.get((drone_id, missile_id))
                if not result or result['params'] is None: continue

                P_D0 = DRONES_INFO[drone_id]['P0']
                params = result['params']
                theta, v, t1, dt1, dt2, f1, f2, f3 = params
                v_vec = np.array([v * np.cos(theta), v * np.sin(theta), 0])
                t_drops = [t1, t1 + dt1, t1 + dt1 + dt2]
                f_times = [f1, f2, f3]
                
                individual_details = result.get('individual_bomb_details', [])

                for i in range(3):
                    p_drop = P_D0 + v_vec * t_drops[i]
                    p_det = p_drop + v_vec * f_times[i] + 0.5 * G * f_times[i]**2
                    
                    single_bomb_duration = 0.0
                    single_bomb_intervals_str = "无"
                    if i < len(individual_details):
                        detail = individual_details[i]
                        single_bomb_duration = detail.get('duration', 0.0)
                        intervals = detail.get('intervals', [])
                        if intervals:
                            single_bomb_intervals_str = ', '.join([f"[{s:.2f}, {e:.2f}]" for s, e in intervals])

                    final_output_data.append({
                        '干扰的导弹编号': missile_id,
                        '无人机编号': drone_id,
                        '无人机运动方向(度)': np.rad2deg(theta),
                        '无人机运动速度(m/s)': v,
                        '投放1(m)': t_drops[0], '投放2': t_drops[1], '投放3': t_drops[2],
                        '起爆1(m)': f_times[0], '起爆2': f_times[1], '起爆3': f_times[2],
                        '烟幕干扰弹编号': f'弹_{i+1}',
                        '投放点X(m)': p_drop[0], '投放点Y(m)': p_drop[1], '投放点Z(m)': p_drop[2],
                        '起爆点X(m)': p_det[0], '起爆点Y(m)': p_det[1], '起爆点Z(m)': p_det[2],
                        '单弹干扰时长(s)': single_bomb_duration,
                        '单弹干扰时间段': single_bomb_intervals_str,
                        '该导弹总有效时长(s)': total_missile_time,
                        '该导弹有效干扰时间段': intervals_str,
                    })

        df_final = pd.DataFrame(final_output_data)
        
        cols_to_clear_missile = ['该导弹总有效时长(s)', '该导弹有效干扰时间段']
        df_final.loc[df_final.duplicated(subset=['干扰的导弹编号']), cols_to_clear_missile] = ""

        cols_to_clear_drone = ['无人机运动方向(度)', '无人机运动速度(m/s)']
        df_final.loc[df_final.duplicated(subset=['干扰的导弹编号', '无人机编号']), cols_to_clear_drone] = ""
        
        df_final.replace("", np.nan, inplace=True)

        print(df_final.to_string(index=False, float_format="%.2f", na_rep=""))

    end_time_total = time.time()
    print("\n" + "="*70)
    print(f"问题5(大规模种群)总耗时: {end_time_total - start_time_total:.2f} 秒")
    print("="*70)