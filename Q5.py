import numpy as np
from scipy.optimize import brentq, differential_evolution
import warnings
import time
import pandas as pd
from itertools import product

# --- 0. 环境设置 ---
# 忽略计算中可能出现的无效值警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 设置Pandas显示格式，方便终端查看
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# --- 1. 定义常量与初始条件 (源自题目) ---

# 保护目标属性 (圆柱体)
R_T = 7.0  # 半径
H_T = 10.0 # 高度
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0]) # 底面圆心

# 烟幕云团属性
R_CLOUD = 10.0  # 烟幕云团有效半径
V_SINK = np.array([0.0, 0.0, -3.0])  # 烟幕云团下沉速度
T_EFFECT = 20.0  # 单个云团有效遮蔽持续时间

# 物理常量
G = np.array([0.0, 0.0, -9.8]) # 重力加速度

# 导弹初始信息
MISSILES_INFO = {
    'M1': {'P0': np.array([20000.0, 0.0, 2000.0])},
    'M2': {'P0': np.array([19000.0, 600.0, 2100.0])},
    'M3': {'P0': np.array([18000.0, -600.0, 1900.0])}
}

# 无人机初始信息
DRONES_INFO = {
    'FY1': {'P0': np.array([17800.0, 0.0, 1800.0])},
    'FY2': {'P0': np.array([12000.0, 1400.0, 1400.0])},
    'FY3': {'P0': np.array([6000.0, -3000.0, 700.0])},
    'FY4': {'P0': np.array([11000.0, 2000.0, 1800.0])},
    'FY5': {'P0': np.array([13000.0, -2000.0, 1300.0])}
}

# 预计算所有导弹的速度矢量
V_M_SCALAR = 300.0
FAKE_TARGET_POS = np.array([0.0, 0.0, 0.0])
for m_id, m_data in MISSILES_INFO.items():
    u_M = FAKE_TARGET_POS - m_data['P0']
    m_data['V_VEC'] = V_M_SCALAR * u_M / np.linalg.norm(u_M)

# --- 2. 核心计算与优化函数 ---

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

# MODIFICATION START: 增加 bomb_indices 参数
def calculate_obscuration_details(params, P_D0, P_M0, v_m_vec, bomb_indices=None):
    """
    计算给定策略参数下的总遮蔽时间及具体的遮蔽时间段。
    此函数是计算的核心，被优化器反复调用。
    
    新增参数:
    bomb_indices (list or None): 一个包含要计算的炸弹索引的列表(例如 [0] 或 [0, 1, 2])。
                                如果为 None, 则计算所有炸弹的联合效果。
    
    返回: (总遮蔽时长, 遮蔽时间段列表)
    """
# MODIFICATION END
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

    # MODIFICATION START: 根据 bomb_indices 筛选要处理的烟幕弹
    if bomb_indices is not None:
        bombs_to_process = [all_bombs_info[i] for i in bomb_indices if i < len(all_bombs_info)]
    else:
        bombs_to_process = all_bombs_info
    # MODIFICATION END

    def get_missile_pos(t): return P_M0 + v_m_vec * t
    def get_cloud_pos(t, bomb_info): return bomb_info['p_det'] + V_SINK * (t - bomb_info['t_start_effect'])
    
    def phi(t):
        p_m = get_missile_pos(t)
        max_m_value = -1.0 # 使用-1代表未遮蔽
        is_obscured = False
        # MODIFICATION START: 使用筛选后的 bombs_to_process
        for bomb in bombs_to_process:
        # MODIFICATION END
            if not (bomb['t_start_effect'] <= t <= bomb['t_end_effect']): continue
            p_cloud = get_cloud_pos(t, bomb)
            A = p_cloud - p_m
            A_mag_sq = np.dot(A, A)
            if A_mag_sq < R_CLOUD**2: # 导弹在烟幕内，完全遮蔽
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
        
        if is_obscured: return 1.0 # 在烟幕内
        return max_m_value

    all_event_times = set()
    # MODIFICATION START: 使用筛选后的 bombs_to_process
    for bomb in bombs_to_process:
    # MODIFICATION END
        all_event_times.add(bomb['t_start_effect'])
        all_event_times.add(bomb['t_end_effect'])
    
    if not all_event_times: return 0.0, []
    
    t_scan_start, t_scan_end = min(all_event_times), max(all_event_times)
    roots = []
    scan_step = 0.1 # 扫描步长，用于寻找根区间
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
        if abs(t1 - t2) < 1e-6: continue # 忽略过短的时间段
        t_mid = (t1 + t2) / 2.0
        if phi(t_mid) >= 0:
            duration = t2 - t1
            total_obscured_time += duration
            # 合并连续的时间段
            if obscured_intervals and abs(obscured_intervals[-1][1] - t1) < 1e-6:
                obscured_intervals[-1][1] = t2
            else:
                obscured_intervals.append([t1, t2])
                
    return total_obscured_time, obscured_intervals

def objective_function_for_pair(params, P_D0, P_M0, v_m_vec):
    """
    针对特定“无人机-导弹”对的优化目标函数。
    在计算前检查动态物理约束（时空剪枝）。
    """
    theta, v_uav, t_drop1, delta_t1, delta_t2, t_fuze1, t_fuze2, t_fuze3 = params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    
    t_drops = [t_drop1, t_drop1 + delta_t1, t_drop1 + delta_t1 + delta_t2]
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    
    for i in range(3):
        t_drop = t_drops[i]
        t_det = t_drop + fuze_times[i]
        
        missile_x_at_det = P_M0[0] + v_m_vec[0] * t_det
        drone_x_at_det = P_D0[0] + v_uav_vec[0] * t_drop
        
        if missile_x_at_det <= drone_x_at_det + 10:
            return 1e9

    total_time, _ = calculate_obscuration_details(params, P_D0, P_M0, v_m_vec)
    return -total_time

def find_best_strategy_for_pair(drone_id, missile_id):
    """
    为指定的无人机和导弹组合，运行优化算法，找到最佳干扰策略。
    """
    drone = DRONES_INFO[drone_id]
    missile = MISSILES_INFO[missile_id]
    P_D0, P_M0, v_m_vec = drone['P0'], missile['P0'], missile['V_VEC']

    print(f"\n--- 正在为组合 [无人机 {drone_id} -> 导弹 {missile_id}] 进行优化... ---")
    
    v_uav_worst_case_x = 140.0
    if (v_m_vec[0] - v_uav_worst_case_x) >= 0:
        time_limit_conservative = 100
    else:
        time_limit_conservative = (P_D0[0] - P_M0[0]) / (v_m_vec[0] - v_uav_worst_case_x)

    t_drop1_max = max(5, time_limit_conservative - 5)
    t_fuze_max = max(5, time_limit_conservative)

    bounds = [
        (-np.pi, np.pi), (70, 140), (0.1, t_drop1_max), (1.0, 5.0),
        (1.0, 5.0), (1.0, t_fuze_max), (1.0, t_fuze_max), (1.0, t_fuze_max)
    ]

    result = differential_evolution(
        objective_function_for_pair, bounds, args=(P_D0, P_M0, v_m_vec),
        strategy='best1bin', maxiter=20, popsize=50, tol=0.01,
        mutation=(0.5, 1), recombination=0.7, disp=True, workers=-1
    )
    
    max_time = -result.fun
    best_params = result.x
    
    print(f"组合 [{drone_id} -> {missile_id}] 优化完成。最长遮蔽时间: {max_time:.4f} 秒")
    
    if max_time <= 0:
        return {
            'drone_id': drone_id, 'missile_id': missile_id,
            'max_time': 0, 'params': None, 'intervals': [],
            # MODIFICATION START: 确保在无效时也返回空列表
            'individual_bomb_details': []
            # MODIFICATION END
        }

    _, intervals = calculate_obscuration_details(best_params, P_D0, P_M0, v_m_vec)
    
    # MODIFICATION START: 计算并存储每个烟幕弹的独立干扰效果
    individual_bomb_details = []
    for i in range(3):
        # 调用改造后的函数，只计算第 i 枚弹的效果
        duration, bomb_intervals = calculate_obscuration_details(best_params, P_D0, P_M0, v_m_vec, bomb_indices=[i])
        individual_bomb_details.append({
            'duration': duration,
            'intervals': bomb_intervals
        })
    # MODIFICATION END

    return {
        'drone_id': drone_id, 'missile_id': missile_id,
        'max_time': max_time, 'params': best_params, 'intervals': intervals,
        # MODIFICATION START: 将独立干扰信息添加到返回结果中
        'individual_bomb_details': individual_bomb_details
        # MODIFICATION END
    }

def calculate_union_of_intervals(intervals_list):
    """计算时间段并集的总长度和合并后的时间段"""
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
    
    print("="*60 + "\n           问题5：最优烟幕投放策略 - 预计算阶段           \n" + "="*60)
    print("说明: 此阶段将为每个'无人机-导弹'组合寻找最优干扰方案。\n该过程计算量巨大，请耐心等待...")

    all_pairs_results = {}
    drone_ids = list(DRONES_INFO.keys())
    missile_ids = list(MISSILES_INFO.keys())

    for d_id in drone_ids:
        for m_id in missile_ids:
            all_pairs_results[(d_id, m_id)] = find_best_strategy_for_pair(d_id, m_id)

    print("\n" + "="*60 + "\n            问题5：最优烟幕投放策略 - 分配阶段            \n" + "="*60)

    best_assignment = None
    max_total_time = -1
    assignments = product(missile_ids, repeat=len(drone_ids))

    for assignment_tuple in assignments:
        current_assignment = {m_id: [] for m_id in missile_ids}
        for i, m_id in enumerate(assignment_tuple):
            current_assignment[m_id].append(drone_ids[i])
        
        if not all(current_assignment.values()): continue
            
        current_total_time = 0
        for m_id, assigned_drones in current_assignment.items():
            intervals_for_missile = [all_pairs_results[(d_id, m_id)]['intervals'] for d_id in assigned_drones]
            union_time, _ = calculate_union_of_intervals(intervals_for_missile)
            current_total_time += union_time
            
        if current_total_time > max_total_time:
            max_total_time = current_total_time
            best_assignment = current_assignment

    print("\n" + "="*70 + "\n              问题5：最终最优烟幕投放策略与结果              \n" + "="*70)
    
    if not best_assignment:
        print("错误：未能找到任何有效的分配方案。")
    else:
        print(f"\n最优分配方案下的最大总遮蔽时间: {max_total_time:.4f} 秒\n")
        final_output_data = []

        for missile_id, assigned_drones in best_assignment.items():
            if not assigned_drones: continue

            intervals_for_missile = [all_pairs_results[(d_id, missile_id)]['intervals'] for d_id in assigned_drones]
            total_missile_time, merged_intervals = calculate_union_of_intervals(intervals_for_missile)
            intervals_str = ', '.join([f"[{start:.2f}s, {end:.2f}s]" for start, end in merged_intervals])
            
            for drone_id in assigned_drones:
                result = all_pairs_results[(drone_id, missile_id)]
                if not result or result['params'] is None: continue

                P_D0 = DRONES_INFO[drone_id]['P0']
                params = result['params']
                theta, v, t1, dt1, dt2, f1, f2, f3 = params
                v_vec = np.array([v * np.cos(theta), v * np.sin(theta), 0])
                t_drops = [t1, t1 + dt1, t1 + dt1 + dt2]
                f_times = [f1, f2, f3]
                
                # MODIFICATION START: 获取单弹干扰详情
                individual_details = result.get('individual_bomb_details', [])
                # MODIFICATION END

                for i in range(3):
                    p_drop = P_D0 + v_vec * t_drops[i]
                    p_det = p_drop + v_vec * f_times[i] + 0.5 * G * f_times[i]**2
                    
                    # MODIFICATION START: 准备单弹干扰的输出信息
                    single_bomb_duration = 0.0
                    single_bomb_intervals_str = "无"
                    if i < len(individual_details):
                        detail = individual_details[i]
                        single_bomb_duration = detail.get('duration', 0.0)
                        intervals = detail.get('intervals', [])
                        if intervals:
                            single_bomb_intervals_str = ', '.join([f"[{s:.2f}, {e:.2f}]" for s, e in intervals])
                    # MODIFICATION END

                    final_output_data.append({
                        '干扰的导弹编号': missile_id,
                        '无人机编号': drone_id,
                        '无人机运动方向(度)': np.rad2deg(theta),
                        '无人机运动速度(m/s)': v,
                        '烟幕干扰弹编号': f'弹_{i+1}',
                        '投放点X(m)': p_drop[0], '投放点Y(m)': p_drop[1], '投放点Z(m)': p_drop[2],
                        '起爆点X(m)': p_det[0], '起爆点Y(m)': p_det[1], '起爆点Z(m)': p_det[2],
                        # MODIFICATION START: 增加新列到字典
                        '单弹干扰时长(s)': single_bomb_duration,
                        '单弹干扰时间段': single_bomb_intervals_str,
                        # MODIFICATION END
                        '该导弹总有效时长(s)': total_missile_time,
                        '该导弹有效干扰时间段': intervals_str,
                    })

        df_final = pd.DataFrame(final_output_data)
        
        cols_to_clear_missile = ['该导弹总有效时长(s)', '该导弹有效干扰时间段']
        df_final.loc[df_final.duplicated(subset=['干扰的导弹编号']), cols_to_clear_missile] = ""

        cols_to_clear_drone = ['无人机运动方向(度)', '无人机运动速度(m/s)']
        df_final.loc[df_final.duplicated(subset=['干扰的导弹编号', '无人机编号']), cols_to_clear_drone] = ""
        
        # 将空字符串替换为 np.nan 以便 float_format 生效
        df_final.replace("", np.nan, inplace=True)

        print(df_final.to_string(index=False, float_format="%.2f", na_rep=""))

    end_time_total = time.time()
    print("\n" + "="*70)
    print(f"问题5求解总耗时: {end_time_total - start_time_total:.2f} 秒")
    print("="*70)