import numpy as np
from scipy.optimize import minimize, Bounds, brentq
from scipy.stats import qmc
import warnings
import time
from tqdm import tqdm

# --- MENTOR'S MODIFICATION: 引入Numba库 ---
import numba
from numba import jit
# --- END OF MODIFICATION ---


# --- MENTOR'S INSIGHT ---
# 模型版本: V4.4 (问题2求解器 - Numba JIT 加速版)
# 核心策略：对计算密集型函数 phi 和 generate_smart_initial_points 
#          应用Numba的JIT编译，将其转换为高速机器码，大幅提升计算效率。

warnings.filterwarnings("ignore")

# --- [常量定义部分与之前版本相同] ---
P_M0 = np.array([20000.0, 0.0, 2000.0]); V_M_SCALAR = 300.0
P_D0 = np.array([17800.0, 0.0, 1800.0]);
R_CLOUD = 10.0; V_SINK = np.array([0.0, 0.0, -3.0]); T_EFFECT = 20.0
R_T = 7.0; H_T = 10.0
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0]); P_T_TOP_CENTER = np.array([0.0, 200.0, 10.0]);
G = np.array([0.0, 0.0, -9.8]);

# --- 2. 目标函数封装 ---
def generate_target_points(num_angles=8):
    points = []; heights = [0, H_T / 2, H_T]; center_xy = P_T_BOTTOM_CENTER[:2]
    for h in heights:
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles); dx = R_T * np.cos(angle); dy = R_T * np.sin(angle)
            points.append(np.array([center_xy[0] + dx, center_xy[1] + dy, h]))
    return np.array(points) # 返回Numpy数组以更好地兼容Numba

TARGET_CHECK_POINTS_24 = generate_target_points(num_angles=8)


# --- MENTOR'S MODIFICATION: 对phi函数进行JIT编译 ---
@jit(nopython=True, cache=True)
def phi_accelerated(t, t_start_effect, t_end_effect, p_m_t, p_c_t, check_points):
    """
    这是一个被Numba加速的、独立的phi函数版本。
    所有外部变量都作为参数传入，以实现纯粹的数值计算。
    """
    if not (t_start_effect <= t <= t_end_effect): return -1.0
    
    A = p_c_t - p_m_t
    A_mag_sq = np.dot(A, A)
    if A_mag_sq < R_CLOUD**2: return 1.0
    
    A_mag = np.sqrt(A_mag_sq)
    cos_cone = np.sqrt(1.0 - R_CLOUD**2 / A_mag_sq)
    
    min_m_value = np.inf
    for i in range(check_points.shape[0]):
        point = check_points[i]
        vec = point - p_m_t
        vec_mag = np.linalg.norm(vec)
        if vec_mag < 1e-9: cos_theta = 1.0
        else: cos_theta = np.dot(vec, A) / (vec_mag * A_mag)
        m = cos_theta - cos_cone
        if m < min_m_value: min_m_value = m
    return min_m_value

def calculate_obscuration_time(params):
    theta, v_uav, t_drop, t_fuze = params
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    p_drop = P_D0 + v_uav_vec * t_drop
    t_det = t_drop + t_fuze
    p_det = p_drop + v_uav_vec * t_fuze + 0.5 * G * t_fuze**2
    t_start_effect = t_det
    t_end_effect = t_det + T_EFFECT
    u_M = np.array([0.0, 0.0, 0.0]) - P_M0
    v_m_vec = V_M_SCALAR * u_M / np.linalg.norm(u_M)

    # 根函数，用于brentq求解
    def root_function(t):
        p_m_t = P_M0 + v_m_vec * t
        p_c_t = p_det + V_SINK * (t - t_det)
        return phi_accelerated(t, t_start_effect, t_end_effect, p_m_t, p_c_t, TARGET_CHECK_POINTS_24)

    # 求解过程
    roots = []; scan_step = 0.1; t_current = t_start_effect
    while t_current < t_end_effect:
        t_next = min(t_current + scan_step, t_end_effect)
        try:
            if root_function(t_current) * root_function(t_next) < 0:
                root = brentq(root_function, t_current, t_next)
                roots.append(root)
        except (ValueError, RuntimeError): pass
        if t_next == t_end_effect: break
        t_current = t_next
        
    events = sorted(list(set([t_start_effect] + roots + [t_end_effect])))
    total_obscured_time = 0.0
    for i in range(len(events) - 1):
        t1, t2 = events[i], events[i+1]
        if abs(t1 - t2) < 1e-6: continue
        if root_function((t1 + t2) / 2.0) >= 0:
            total_obscured_time += (t2 - t1)
    return total_obscured_time

def objective_function(params):
    return -calculate_obscuration_time(params)

# --- MENTOR'S MODIFICATION: 对智能采样函数进行JIT编译 ---
@jit(nopython=True, cache=True)
def generate_smart_initial_points_accelerated(n_points, v_uav_bounds_min, v_uav_bounds_max):
    initial_points = np.zeros((n_points, 4)) # 预分配内存
    count = 0
    
    # 将外部常量传入
    _P_D0 = P_D0
    _G = G
    _P_M0 = P_M0
    _V_M_SCALAR = V_M_SCALAR
    _P_T_TOP_CENTER = P_T_TOP_CENTER
    
    u_M = np.array([0.0, 0.0, 0.0]) - _P_M0
    v_m_vec = _V_M_SCALAR * u_M / np.linalg.norm(u_M)

    while count < n_points:
        t_int = np.random.uniform(55, 65)
        t_det = t_int
        p_missile_at_int = _P_M0 + v_m_vec * t_int
        los_vec = _P_T_TOP_CENTER - p_missile_at_int
        p_det_ideal = p_missile_at_int + np.random.uniform(0.3, 0.7) * los_vec
        t_fuze = np.random.uniform(2, 8)
        
        if t_det - t_fuze <= 0: continue
        
        v_uav_vec_required = (p_det_ideal - _P_D0 - 0.5 * _G * t_fuze**2) / t_det
        v_uav_vec_required[2] = 0
        v_uav_scalar = np.linalg.norm(v_uav_vec_required)
        
        if not (v_uav_bounds_min <= v_uav_scalar <= v_uav_bounds_max): continue
        
        theta = np.arctan2(v_uav_vec_required[1], v_uav_vec_required[0])
        if theta < 0: theta += 2 * np.pi
        t_drop = t_det - t_fuze
        
        initial_points[count, 0] = theta
        initial_points[count, 1] = v_uav_scalar
        initial_points[count, 2] = t_drop
        initial_points[count, 3] = t_fuze
        count += 1
        
    return initial_points

if __name__ == "__main__":
    print("--- 问题2求解开始 (模型版本: V4.4 - Numba JIT 加速版) ---")
    bounds_list = [(0, 2*np.pi), (70, 140), (0, 70), (1, 10)]
    bounds_obj = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])
    
    # 第1阶段：生成智能初始点 (调用加速版函数)
    N_STARTS = 100
    start_time = time.time()
    initial_points = generate_smart_initial_points_accelerated(N_STARTS, bounds_list[1][0], bounds_list[1][1])
    print(f"Phase 1: 生成 {N_STARTS} 个智能初始点耗时: {time.time() - start_time:.4f} 秒")
    
    # 第2阶段：局部精炼
    best_fun = float('inf')
    best_x = None
    print("\nPhase 2: 从智能初始点开始进行局部精炼...")
    start_time = time.time()
    for i in tqdm(range(N_STARTS), desc="局部精炼进度"):
        x0 = initial_points[i]
        res = minimize(
            objective_function, x0, method='L-BFGS-B', bounds=bounds_obj,
            options={'ftol': 1e-6, 'maxiter': 200}
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    print(f"Phase 2: 局部精炼耗时: {time.time() - start_time:.4f} 秒")

    # 结果展示... (与之前版本相同)
    max_time = -best_fun; best_params = best_x
    print("\n" + "="*50)
    print("          问题2：最优烟幕投放策略 (Numba加速求解)          ")
    print("="*50)
    # ... (省略与之前相同的详细打印)
    print(f"\n找到的最优策略可获得最长有效遮蔽时间: {max_time:.4f} 秒")
    print(f"无人机飞行方向 (角度): {np.rad2deg(best_params[0]):.2f} 度, 速度: {best_params[1]:.2f} m/s")
    print(f"投放时间: {best_params[2]:.2f} s, 引信时间: {best_params[3]:.2f} s")
# 找到的最优策略可获得最长有效遮蔽时间: 4.5414 秒
# --- 最优决策变量 ---
# 无人机飞行方向 (角度): 176.91 度
# 无人机飞行速度: 74.24 m/s
# 烟幕弹投放时间: 0.00 s
# 烟幕弹引信时间: 2.56 s

# --- 对应的策略结果 ---
# 烟幕弹投放点坐标: (17800.00, 0.00, 1800.00)
# 烟幕弹起爆点坐标: (17610.44, 10.23, 1767.96)