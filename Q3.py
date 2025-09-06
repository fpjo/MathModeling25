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

# 计算导弹的运动轨迹
u_M = np.array([0.0, 0.0, 0.0]) - P_M0 # 指向假目标(原点)
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

# 生成24个目标检查点
TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)

def calculate_total_obscuration_time_multi(params):
    """
    计算三枚烟幕弹联合作用下的总有效遮蔽时间。
    参数 (params): 一个包含6个决策变量的列表或数组
        - params[0]: theta (无人机飞行方向, 弧度)
        - params[1]: v_uav (无人机飞行速度, m/s)
        - params[2]: t_drop1 (第一枚弹的投放时间, s)
        - params[3]: t_fuze1 (第一枚弹的引信时间, s)
        - params[4]: t_fuze2 (第二枚弹的引信时间, s)
        - params[5]: t_fuze3 (第三枚弹的引信时间, s)
    """
    theta, v_uav, t_drop1, t_fuze1, t_fuze2, t_fuze3 = params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])

    # 计算三枚弹的投放和起爆信息
    bombs_info = []
    fuze_times = [t_fuze1, t_fuze2, t_fuze3]
    for i in range(3):
        t_drop = t_drop1 + i  # 投放间隔为1s
        t_fuze = fuze_times[i]
        
        p_drop = P_D0 + v_uav_vec * t_drop
        t_det = t_drop + t_fuze
        p_det = p_drop + v_uav_vec * t_fuze + 0.5 * G * t_fuze**2
        
        bombs_info.append({
            'p_det': p_det,
            't_start_effect': t_det,
            't_end_effect': t_det + T_EFFECT,
        })


    def get_missile_pos(t):
        return P_M0 + v_m_vec * t

    def get_cloud_pos(t, bomb_info):
        return bomb_info['p_det'] + V_SINK * (t - bomb_info['t_start_effect'])

    def phi(t):
        """
        判断在t时刻，目标是否被任意一个烟幕云团遮蔽。
        返回值为 max(m_i)，如果 max(m_i) >= 0, 则表示被遮蔽。
        """
        p_m = get_missile_pos(t)
        max_m_value = -1.0 # 默认为未遮蔽

        for bomb in bombs_info:
            # 检查t是否在该云团的有效时间内
            if not (bomb['t_start_effect'] <= t <= bomb['t_end_effect']):
                continue

            p_cloud = get_cloud_pos(t, bomb)
            A = p_cloud - p_m
            A_mag_sq = np.dot(A, A)

            # 如果导弹在云团内部，则所有目标点都被遮蔽
            if A_mag_sq < R_CLOUD**2:
                return 1.0

            A_mag = np.sqrt(A_mag_sq)
            # 计算遮蔽锥的半顶角余弦值
            cos_cone = np.sqrt(1 - R_CLOUD**2 / A_mag_sq)
            
            min_m_value_for_this_cloud = float('inf')
            # 检查所有目标点
            for point in TARGET_CHECK_POINTS_24:
                vec = point - p_m
                vec_mag = np.linalg.norm(vec)
                
                if vec_mag < 1e-9: # 避免除以零
                    cos_theta = 1.0
                else:
                    cos_theta = np.dot(vec, A) / (vec_mag * A_mag)
                
                m = cos_theta - cos_cone
                if m < min_m_value_for_this_cloud:
                    min_m_value_for_this_cloud = m
            
            # 更新全局最大的m值
            if min_m_value_for_this_cloud > max_m_value:
                max_m_value = min_m_value_for_this_cloud
        
        return max_m_value

    # 确定需要扫描的时间区间和关键事件点
    all_event_times = set()
    for bomb in bombs_info:
        all_event_times.add(bomb['t_start_effect'])
        all_event_times.add(bomb['t_end_effect'])
    
    if not all_event_times: return 0.0
    
    t_scan_start = min(all_event_times)
    t_scan_end = max(all_event_times)
    
    # 使用brentq方法寻找 phi(t)=0 的根，以精确计算遮蔽区间的边界
    roots = []
    scan_step = 0.1 # 扫描步长
    t_current = t_scan_start
    while t_current < t_scan_end:
        t_next = min(t_current + scan_step, t_scan_end)
        try:
            # 如果函数值在区间两端异号，则存在根
            if phi(t_current) * phi(t_next) < 0:
                root = brentq(phi, t_current, t_next)
                roots.append(root)
        except (ValueError, RuntimeError):
            pass
        if t_next == t_scan_end:
            break
        t_current = t_next
        
    # 将所有事件点（区间起止点和根）排序
    events = sorted(list(all_event_times.union(roots)))
    
    total_obscured_time = 0.0
    # 检查每个小区间的中点，如果中点被遮蔽，则累加该区间的时长
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue # 忽略过小的区间
        
        mid_point_time = (t1 + t2) / 2.0
        if phi(mid_point_time) >= 0:
            total_obscured_time += (t2 - t1)
            
    return total_obscured_time

def objective_function_multi(params):
    """优化器调用的目标函数，返回负的总遮蔽时间（因为优化器是最小化）"""
    return -calculate_total_obscuration_time_multi(params)

# --- 3. 执行优化求解 ---
if __name__ == "__main__":
    print("--- 问题3求解开始 ---")
    print("正在使用差分进化算法寻找3枚干扰弹的最优投放策略...")
    print("--- 问题3求解开始 (保守Bounds + 动态约束惩罚) ---")
    
    # --- 角度搜索范围设定 ---
    P_D0_XY = P_D0[:2]; P_T_CENTER_XY = P_T_BOTTOM_CENTER[:2]
    vec_to_target = P_T_CENTER_XY - P_D0_XY
    angle_to_target_rad = np.arctan2(vec_to_target[1], vec_to_target[0])
    lower_angle_bound_rad = angle_to_target_rad
    upper_angle_bound_rad = np.deg2rad(180)

    # --- 计算并设定保守的时间Bounds ---
    # 基于最慢无人机速度(70m/s)计算出的最早“拦截失效时间”
    v_uav_worst_case = 70 * np.cos(upper_angle_bound_rad)
    time_limit_conservative = (P_D0[0] - P_M0[0] + 10) / (v_m_vec[0] - v_uav_worst_case)

    # 根据保守时间上限，设定更紧的搜索边界
    t_drop1_max = time_limit_conservative - 2
    t_fuze_max = time_limit_conservative - 2

    print("\n--- 优化参数设定 ---")
    print(f"飞行角度搜索范围: [{np.rad2deg(lower_angle_bound_rad):.2f}, {np.rad2deg(upper_angle_bound_rad):.2f}] 度")
    print(f"保守的拦截时机上限: {time_limit_conservative:.2f} 秒")
    print(f"投放时间 t_drop1 上限收紧至: {t_drop1_max:.2f} 秒")
    print(f"引信时间 t_fuze 上限收紧至: {t_fuze_max:.2f} 秒")

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
    # 注意: maxiter和popsize是关键参数，增加它们会提高精度但消耗更多时间
    # 为了在合理时间内得到结果，这里设置了相对适中的值
    result = differential_evolution(
        objective_function_multi, 
        bounds, 
        strategy='best1bin', 
        maxiter=100,      # 迭代次数
        popsize=300,       # 种群大小
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

    # 准备用于生成Excel的数据
    results_data = []
    fuze_times_opt = [t_fuze1_opt, t_fuze2_opt, t_fuze3_opt]
    
    print("\n--- 各干扰弹投放详情 ---")
    for i in range(3):
        t_drop_opt = t_drop1_opt + i
        t_fuze_opt = fuze_times_opt[i]

        p_drop_opt = P_D0 + v_uav_vec_opt * t_drop_opt
        p_det_opt = p_drop_opt + v_uav_vec_opt * t_fuze_opt + 0.5 * G * t_fuze_opt**2
        
        print(f"\n* 干扰弹 {i+1}:")
        print(f"  - 投放时间: {t_drop_opt:.2f} s")
        print(f"  - 引信时间: {t_fuze_opt:.2f} s")
        print(f"  - 投放点坐标: ({p_drop_opt[0]:.2f}, {p_drop_opt[1]:.2f}, {p_drop_opt[2]:.2f})")
        print(f"  - 起爆点坐标: ({p_det_opt[0]:.2f}, {p_det_opt[1]:.2f}, {p_det_opt[2]:.2f})")
        
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

    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(results_data)
    
    # 添加无人机飞行策略和总遮蔽时间等概要信息
    summary_df = pd.DataFrame({
        '无人机飞行方向(度)': [np.rad2deg(theta_opt)],
        '无人机飞行速度(m/s)': [v_uav_opt],
        '最长总遮蔽时间(s)': [max_time]
    })
    
    # 输出到终端，而不保存为 Excel
    print("\n" + "="*60)
    print("概要与结果")
    print(summary_df.to_string(index=False, float_format="%.4f"))
    print("\n各干扰弹投放详情")
    print(df.to_string(index=False, float_format="%.4f"))
    print("="*60)