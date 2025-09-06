import numpy as np
from scipy.optimize import brentq, differential_evolution
import warnings
import time
import pandas as pd
from itertools import permutations
import os

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

def calculate_obscuration_details(params, P_D0, P_M0, v_m_vec):
    """
    计算给定策略参数下的总遮蔽时间及具体的遮蔽时间段。
    此函数是计算的核心，被优化器反复调用。
    
    返回: (总遮蔽时长, 遮蔽时间段列表)
    """
    theta, v_uav, t_drop1, delta_t1, delta_t2, t_fuze1, t_fuze2, t_fuze3 = params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])

    t_drops = [t_drop1, t_drop1 + delta_t1, t_drop1 + delta_t1 + delta_t2]
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    
    bombs_info = []
    for i in range(3):
        t_drop = t_drops[i]
        t_fuze = fuze_times[i]
        t_det = t_drop + t_fuze
        p_drop = P_D0 + v_uav_vec * t_drop
        p_det = p_drop + v_uav_vec * t_fuze + 0.5 * G * t_fuze**2
        
        bombs_info.append({
            'p_drop': p_drop, 'p_det': p_det,
            't_start_effect': t_det, 't_end_effect': t_det + T_EFFECT,
        })
    
    def get_missile_pos(t): return P_M0 + v_m_vec * t
    def get_cloud_pos(t, bomb_info): return bomb_info['p_det'] + V_SINK * (t - bomb_info['t_start_effect'])
    
    def phi(t):
        p_m = get_missile_pos(t)
        max_m_value = -1.0 # 使用-1代表未遮蔽
        is_obscured = False
        for bomb in bombs_info:
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
    for bomb in bombs_info:
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
        
        # 时空剪枝：确保烟幕在导弹到达前形成
        # 简化为比较x坐标，假设导弹主要沿x负方向飞行
        missile_x_at_det = P_M0[0] + v_m_vec[0] * t_det
        drone_x_at_det = P_D0[0] + v_uav_vec[0] * t_drop # 无人机在投放时刻的位置
        
        # 物理约束：起爆点必须在导弹前方。这里留有10m余量。
        if missile_x_at_det <= drone_x_at_det + 10:
            return 1e9  # 返回巨大惩罚值，放弃此解

    # 如果所有约束都满足，则正常计算目标函数值（总遮蔽时间）
    # 优化器求最小值，因此返回负值
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

    # --- 时空剪枝：设定安全的搜索边界 ---
    # 计算一个保守的拦截时间上限
    # 考虑无人机以最快速度背离导弹飞行的最坏情况
    v_uav_worst_case_x = 140.0 
    # 确保分母为正
    if (v_m_vec[0] - v_uav_worst_case_x) >= 0:
        time_limit_conservative = 100 # 如果速度差为负或零，给一个默认大值
    else:
        time_limit_conservative = (P_D0[0] - P_M0[0]) / (v_m_vec[0] - v_uav_worst_case_x)

    t_drop1_max = max(5, time_limit_conservative - 5) # 留出余量
    t_fuze_max = max(5, time_limit_conservative)

    # --- 定义优化变量的边界 ---
    # 去掉角度剪枝，允许无人机朝任意方向飞行
    bounds = [
        (-np.pi, np.pi),   # theta: 飞行方向角度 (-180 to 180 degrees)
        (70, 140),         # v_uav: 无人机速度
        (0.1, t_drop1_max),# t_drop1: 第一次投放时间
        (1.0, 5.0),        # delta_t1: 第一次与第二次投放的间隔 (至少1s)
        (1.0, 5.0),        # delta_t2: 第二次与第三次投放的间隔 (至少1s)
        (1.0, t_fuze_max), # t_fuze1: 引信1时间
        (1.0, t_fuze_max), # t_fuze2: 引信2时间
        (1.0, t_fuze_max)  # t_fuze3: 引信3时间
    ]

    # --- 运行差分进化算法 ---
    # 注意：为了演示，maxiter和popsize设置得较小。
    # 在实际竞赛中，应使用更大的值以获得更优结果，但这会显著增加计算时间。
    result = differential_evolution(
        objective_function_for_pair,
        bounds,
        args=(P_D0, P_M0, v_m_vec),
        strategy='best1bin',
        maxiter=50,      # 演示值，建议增加到 200+
        popsize=100,      # 演示值，建议增加到 50+
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,      # 关闭每次迭代的输出
        workers=-1       # 使用所有可用的CPU核心
    )
    
    max_time = -result.fun
    best_params = result.x
    
    print(f"组合 [{drone_id} -> {missile_id}] 优化完成。最长遮蔽时间: {max_time:.4f} 秒")
    
    if max_time <= 0: # 如果找不到有效解
        return {
            'drone_id': drone_id, 'missile_id': missile_id,
            'max_time': 0, 'params': None, 'intervals': []
        }

    # 使用最优参数重新计算详细信息
    _, intervals = calculate_obscuration_details(best_params, P_D0, P_M0, v_m_vec)
    
    return {
        'drone_id': drone_id, 'missile_id': missile_id,
        'max_time': max_time, 'params': best_params, 'intervals': intervals
    }


# --- 3. 主执行流程 ---
if __name__ == "__main__":
    start_time_total = time.time()
    
    # --- STAGE 1: 预计算所有15个组合的最优策略 ---
    print("="*60)
    print("           问题5：最优烟幕投放策略 - 预计算阶段           ")
    print("="*60)
    print("说明: 此阶段将为每个'无人机-导弹'组合寻找最优干扰方案。")
    print("该过程计算量巨大，请耐心等待...")

    all_pairs_results = []
    for d_id in DRONES_INFO.keys():
        for m_id in MISSILES_INFO.keys():
            result = find_best_strategy_for_pair(d_id, m_id)
            all_pairs_results.append(result)

    # --- STAGE 2: 寻找最优的无人机-导弹分配方案 ---
    print("\n" + "="*60)
    print("            问题5：最优烟幕投放策略 - 分配阶段            ")
    print("="*60)
    print("说明: 正在从预计算结果中寻找最佳的无人机任务分配...")

    drone_ids = list(DRONES_INFO.keys())
    missile_ids = list(MISSILES_INFO.keys())
    
    best_assignment = None
    max_total_time = -1

    # 生成所有从5架无人机中选出3架的组合
    drone_permutations = permutations(drone_ids, len(missile_ids))

    # 遍历所有分配可能性
    for drone_assign in drone_permutations:
        current_total_time = 0
        # drone_assign 是一个元组，如 ('FY1', 'FY3', 'FY5')
        # 默认分配给 ('M1', 'M2', 'M3')
        assignment_map = dict(zip(drone_assign, missile_ids))
        
        for d_id, m_id in assignment_map.items():
            # 从预计算结果中查找该组合的得分
            res = next((r for r in all_pairs_results if r['drone_id'] == d_id and r['missile_id'] == m_id), None)
            if res:
                current_total_time += res['max_time']
        
        if current_total_time > max_total_time:
            max_total_time = current_total_time
            best_assignment = assignment_map
    
    print("最优分配方案寻找完毕！")

    # --- STAGE 3: 整理并输出最终结果 ---
    print("\n" + "="*70)
    print("              问题5：最终最优烟幕投放策略与结果              ")
    print("="*70)
    
    if not best_assignment:
        print("错误：未能找到任何有效的分配方案。")
    else:
        print(f"\n最优分配方案下的最大总遮蔽时间: {max_total_time:.4f} 秒\n")
        print("具体分配如下:")
        for d, m in best_assignment.items():
            print(f"  - 无人机 {d}  ->  导弹 {m}")
        
        print("\n" + "-"*70)
        print("各任务单元详细策略信息:")
        print("-"*70)

        final_output_data = []

        for drone_id, missile_id in best_assignment.items():
            # 提取该最优分配的详细结果
            result = next((r for r in all_pairs_results if r['drone_id'] == drone_id and r['missile_id'] == missile_id), None)
            if not result or result['params'] is None:
                continue

            # 解包参数
            P_D0 = DRONES_INFO[drone_id]['P0']
            params = result['params']
            theta_opt, v_uav_opt, t_drop1_opt, delta_t1_opt, delta_t2_opt, t_fuze1_opt, t_fuze2_opt, t_fuze3_opt = params
            v_uav_vec_opt = np.array([v_uav_opt * np.cos(theta_opt), v_uav_opt * np.sin(theta_opt), 0])
            
            t_drops_opt = [t_drop1_opt, t_drop1_opt + delta_t1_opt, t_drop1_opt + delta_t1_opt + delta_t2_opt]
            fuze_times_opt = [t_fuze1_opt, t_fuze2_opt, t_fuze3_opt]
            
            # 格式化时间段输出
            intervals_str = ', '.join([f"[{start:.2f}s, {end:.2f}s]" for start, end in result['intervals']])

            for i in range(3):
                t_drop_opt = t_drops_opt[i]
                t_fuze_opt = fuze_times_opt[i]

                p_drop_opt = P_D0 + v_uav_vec_opt * t_drop_opt
                p_det_opt = p_drop_opt + v_uav_vec_opt * t_fuze_opt + 0.5 * G * t_fuze_opt**2
                
                final_output_data.append({
                    '无人机编号': drone_id,
                    '无人机运动方向(度)': np.rad2deg(theta_opt),
                    '无人机运动速度(m/s)': v_uav_opt,
                    '烟幕干扰弹编号': f'弹_{i+1}',
                    '投放点X(m)': p_drop_opt[0],
                    '投放点Y(m)': p_drop_opt[1],
                    '投放点Z(m)': p_drop_opt[2],
                    '起爆点X(m)': p_det_opt[0],
                    '起爆点Y(m)': p_det_opt[1],
                    '起爆点Z(m)': p_det_opt[2],
                    '对该导弹有效干扰时长(s)': result['max_time'],
                    '有效干扰时间段': intervals_str,
                    '干扰的导弹编号': missile_id
                })

        # 创建DataFrame并进行格式化输出
        df_final = pd.DataFrame(final_output_data)
        
        # 为了输出美观，将只在第一枚弹处显示重复信息
        cols_to_clear = ['无人机运动方向(度)', '无人机运动速度(m/s)', '对该导弹有效干扰时长(s)', '有效干扰时间段', '干扰的导弹编号']
        df_final.loc[df_final['烟幕干扰弹编号'] != '弹_1', cols_to_clear] = ""
        
        print(df_final.to_string(index=False, float_format="%.2f"))

    end_time_total = time.time()
    print("\n" + "="*70)
    print(f"问题5求解总耗时: {end_time_total - start_time_total:.2f} 秒")
    print("="*70)