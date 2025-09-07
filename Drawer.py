import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
import os


# --- 0. 中文字体解决方案 ---
def find_chinese_font():
    """
    在Linux系统中智能查找可用的中文字体。
    """
    font_list = [
        'WenQuanYi Zen Hei',  # 文泉驿正黑 (推荐安装)
        'Noto Sans CJK SC',   # 思源黑体
        'Source Han Sans SC', # 思源黑体
        'AR PL UKai CN',      # 文鼎PL中楷
        'SimHei'              # 黑体 (某些环境可能已安装)
    ]

    for font_name in font_list:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path:
                print(f"[信息] 成功找到中文字体: {font_name} (路径: {font_path})")
                return fm.FontProperties(fname=font_path)
        except Exception:
            continue

    return None


# --- 1. 全局常量定义 ---
R_T = 7.0
H_T = 10.0
P_T_BOTTOM_CENTER = np.array([0.0, 200.0, 0.0])
R_CLOUD = 10.0
G = np.array([0.0, 0.0, -9.8])

MISSILES_INFO = {
    'M1': {'P0': np.array([20000.0, 0.0, 2000.0]), 'color': 'red'},
    'M2': {'P0': np.array([19000.0, 600.0, 2100.0]), 'color': 'darkred'},
    'M3': {'P0': np.array([18000.0, -600.0, 1900.0]), 'color': 'firebrick'}
}

V_M_SCALAR = 300.0

DRONES_INFO = {
    'FY1': {'P0': np.array([17800.0, 0.0, 1800.0]), 'color': 'blue'},
    'FY2': {'P0': np.array([12000.0, 1400.0, 1400.0]), 'color': 'green'},
    'FY3': {'P0': np.array([6000.0, -3000.0, 700.0]), 'color': 'purple'},
    'FY4': {'P0': np.array([11000.0, 2000.0, 1800.0]), 'color': 'orange'},
    'FY5': {'P0': np.array([13000.0, -2000.0, 1300.0]), 'color': 'cyan'}
}

GRENADE_COLORS = ['magenta', 'khaki', 'lime', 'gold', 'aqua']

FAKE_TARGET_POS = np.array([0.0, 0.0, 0.0])

for m_id, m_data in MISSILES_INFO.items():
    u_M = FAKE_TARGET_POS - m_data['P0']
    m_data['V_VEC'] = V_M_SCALAR * u_M / np.linalg.norm(u_M)


# --- 2. 轨迹计算函数 ---
def calculate_missile_trajectory(missile_id, t_end, dt=0.1):
    p0 = MISSILES_INFO[missile_id]['P0']
    v_vec = MISSILES_INFO[missile_id]['V_VEC']
    times = np.arange(0, t_end + dt, dt)
    return p0 + v_vec * times[:, np.newaxis]


def calculate_drone_trajectory(drone_id, drone_params, t_end, dt=0.1):
    p0 = DRONES_INFO[drone_id]['P0']
    theta, v_uav = drone_params['theta'], drone_params['v_uav']
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    times = np.arange(0, t_end + dt, dt)
    return p0 + v_uav_vec * times[:, np.newaxis]


def calculate_grenade_trajectory(drone_id, drone_params, grenade_params, dt=0.05):
    p_drone0 = DRONES_INFO[drone_id]['P0']
    theta, v_uav = drone_params['theta'], drone_params['v_uav']
    t_drop, t_fuze = grenade_params['t_drop'], grenade_params['t_fuze']
    v_uav_vec = np.array([v_uav * np.cos(theta), v_uav * np.sin(theta), 0])
    p_drop = p_drone0 + v_uav_vec * t_drop
    times_after_drop = np.arange(0, t_fuze + dt, dt)
    trajectory = p_drop + v_uav_vec * times_after_drop[:, np.newaxis] + 0.5 * G * times_after_drop[:, np.newaxis]**2
    return trajectory, trajectory[-1]


# --- 3. 绘图辅助函数 ---
def draw_cylinder(ax, radius, height, position):
    z = np.linspace(0, height, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + position[0]
    y_grid = radius * np.sin(theta_grid) + position[1]
    z_grid += position[2]
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='gray')


def draw_sphere(ax, center, radius, color):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.4)


def plot_scenario(missile_ids, strategies, title="三维战场态势示意图", aspect_ratio_factors=(1, 1, 2)):
    fig = plt.figure(figsize=(40, 30))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0], [0], [0], 'k^', markersize=10, label='假目标 (原点)')
    draw_cylinder(ax, R_T, H_T, P_T_BOTTOM_CENTER)
    ax.text(P_T_BOTTOM_CENTER[0], P_T_BOTTOM_CENTER[1], P_T_BOTTOM_CENTER[2] + H_T + 50, "真目标", color='black')

    max_time = 70
    points_of_interest = [P_T_BOTTOM_CENTER, FAKE_TARGET_POS]

    for drone_id, strategy in strategies.items():
        drone_color = DRONES_INFO[drone_id]['color']
        drone_p0 = DRONES_INFO[drone_id]['P0']
        latest_event_time = max([g['t_drop'] + g['t_fuze'] for g in strategy.get('grenades', [])], default=0)
        max_time = max(max_time, latest_event_time)
        drone_traj = calculate_drone_trajectory(drone_id, strategy['params'], latest_event_time)
        ax.plot(drone_traj[:, 0], drone_traj[:, 1], drone_traj[:, 2], color=drone_color, linestyle='--', label=f'无人机 {drone_id} 轨迹')
        ax.plot([drone_p0[0]], [drone_p0[1]], [drone_p0[2]], 'o', color=drone_color, markersize=8, label=f'无人机 {drone_id} 起点')
        points_of_interest.append(drone_p0)

        for i, grenade in enumerate(strategy.get('grenades', [])):
            grenade_traj, p_det = calculate_grenade_trajectory(drone_id, strategy['params'], grenade)
            ax.plot(grenade_traj[:, 0], grenade_traj[:, 1], grenade_traj[:, 2], color=drone_color, linestyle=':', label=f'{drone_id} - 弹{i+1} 轨迹')
            draw_sphere(ax, p_det, R_CLOUD, drone_color)
            # ax.text(p_det[0], p_det[1], p_det[2] + 20, f'{drone_id}-弹{i+1}\n起爆点', color=drone_color)
            points_of_interest.extend(grenade_traj)

    for missile_id in missile_ids:
        missile_color = MISSILES_INFO[missile_id]['color']
        missile_p0 = MISSILES_INFO[missile_id]['P0']
        missile_traj = calculate_missile_trajectory(missile_id, max_time + 5)
        ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], color=missile_color, linewidth=2, label=f'导弹 {missile_id} 轨迹')
        ax.plot([missile_p0[0]], [missile_p0[1]], [missile_p0[2]], 'X', color=missile_color, markersize=10, label=f'导弹 {missile_id} 起点')
        points_of_interest.append(missile_p0)
        points_of_interest.extend([p for p in missile_traj if p[0] < 5000])

    poi_array = np.array(points_of_interest)
    min_coords = poi_array.min(axis=0)
    max_coords = poi_array.max(axis=0)

    x_range = (max_coords[0] - min_coords[0]) * 1.1
    y_range = (max_coords[1] - min_coords[1]) * 1.1
    z_range = (max_coords[2] - min_coords[2]) * 1.1

    min_view_size = 500
    x_range = max(x_range, min_view_size)
    y_range = max(y_range, min_view_size)
    z_range = max(z_range, min_view_size)

    centers = (max_coords + min_coords) / 2

    ax.set_xlim(centers[0] - x_range / 2, centers[0] + x_range / 2)
    ax.set_ylim(centers[1] - y_range / 2, centers[1] + y_range / 2)
    ax.set_zlim(0, max(centers[2] + z_range / 2, H_T + 200))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6))

    x_range_plot = np.ptp(ax.get_xlim())
    y_range_plot = np.ptp(ax.get_ylim())
    z_range_plot = np.ptp(ax.get_zlim())
    ax.set_box_aspect((
        x_range_plot * aspect_ratio_factors[0],
        y_range_plot * aspect_ratio_factors[1],
        z_range_plot * aspect_ratio_factors[2]
    ))

    ax.set_xlabel('X 轴 (m)', fontsize=12)
    ax.set_ylabel('Y 轴 (m)', fontsize=12)
    ax.set_zlabel('Z 轴 (m)', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.view_init(elev=25, azim=-135)
    fig.tight_layout()
    plt.show()
def plot_interception_details(missile_id, drone_id, strategy, title="拦截细节放大图", zoom_range=300):
    """
    绘制指定无人机和导弹在拦截点附近的局部放大图。
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    drone_params = strategy['params']
    drone_color = DRONES_INFO[drone_id]['color']
    missile_color = MISSILES_INFO[missile_id]['color']

    # 绘制无人机、手榴弹和导弹
    grenades = strategy.get('grenades', [])
    if grenades:
        # 找到所有手榴弹的起爆点和最晚事件时间，以确定视图中心和轨迹长度
        all_p_det = []
        max_event_time = 0
        for grenade in grenades:
            _, p_det = calculate_grenade_trajectory(drone_id, drone_params, grenade)
            all_p_det.append(p_det)
            max_event_time = max(max_event_time, grenade['t_drop'] + grenade['t_fuze'])
        
        center_p = np.mean(all_p_det, axis=0)

        for i, grenade in enumerate(grenades):
            grenade_color = GRENADE_COLORS[i % len(GRENADE_COLORS)]
            t_drop, t_fuze = grenade['t_drop'], grenade['t_fuze']
            
            # 计算并绘制手榴弹轨迹和爆炸范围
            grenade_traj, p_det = calculate_grenade_trajectory(drone_id, drone_params, grenade)
            ax.plot(grenade_traj[:, 0], grenade_traj[:, 1], grenade_traj[:, 2], color=grenade_color, linestyle=':', label=f'{drone_id} - 弹{i+1} 轨迹')
            draw_sphere(ax, p_det, R_CLOUD, grenade_color)
            ax.text(p_det[0], p_det[1], p_det[2] + 20, f'弹{i+1}起爆点', color=grenade_color)

            # --- 计算并绘制烟幕运动轨迹 ---
            t_smoke_duration = 20.0
            v_smoke = np.array([0, 0, -3.0])
            p_smoke_end = p_det + v_smoke * t_smoke_duration
            smoke_path = np.array([p_det, p_smoke_end])
            ax.plot(smoke_path[:, 0], smoke_path[:, 1], smoke_path[:, 2], color=grenade_color, linestyle='-.', label=f'弹{i+1}烟幕轨迹')
            draw_sphere(ax, p_smoke_end, R_CLOUD, grenade_color) # 在终点也画一个球
            ax.text(p_smoke_end[0], p_smoke_end[1], p_smoke_end[2] - 30, f'弹{i+1}烟幕终点', color=grenade_color)
            # --- 修改结束 ---

        # 计算并绘制无人机轨迹
        drone_traj = calculate_drone_trajectory(drone_id, drone_params, max_event_time + 1)
        ax.plot(drone_traj[:, 0], drone_traj[:, 1], drone_traj[:, 2], color=drone_color, linestyle='--', label=f'无人机 {drone_id} 轨迹')
        
        # 计算并绘制导弹轨迹 (延长计算时间以覆盖烟幕运动)
        missile_traj = calculate_missile_trajectory(missile_id, max_event_time + t_smoke_duration + 1)
        ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], color=missile_color, linewidth=2, label=f'导弹 {missile_id} 轨迹')

        # 设置坐标轴范围，聚焦于所有爆炸点的中心
        ax.set_xlim(center_p[0] - zoom_range, center_p[0] + zoom_range)
        ax.set_ylim(center_p[1] - zoom_range, center_p[1] + zoom_range)
        ax.set_zlim(center_p[2] - zoom_range, center_p[2] + zoom_range)

    ax.set_xlabel('X 轴 (m)', fontsize=12)
    ax.set_ylabel('Y 轴 (m)', fontsize=12)
    ax.set_zlabel('Z 轴 (m)', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend()
    ax.set_box_aspect((1, 2, 1)) # 保持局部坐标系1:1:1比例
    ax.view_init(elev=0, azim=75)
    fig.tight_layout()
    plt.show()

# --- 5. 主程序入口 ---
if __name__ == '__main__':
    font_prop = find_chinese_font()
    if font_prop:
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        print(f"[信息] Matplotlib 全局中文字体已设置为: {font_prop.get_name()}")
    else:
        print("[警告] 未在您的Ubuntu系统中找到推荐的中文字体。")
        print("[建议] 请安装'文泉驿正黑'字体: sudo apt-get install -y fonts-wqy-zenhei")
        print("[提示] 清除缓存命令: rm -rf ~/.cache/matplotlib")

    plt.rcParams['axes.unicode_minus'] = False
    #问题1绘图
    strategy_q1_example = {
        'FY1': {
            'params': {'theta': np.deg2rad(180), 'v_uav': 120},
            'grenades': [{'t_drop': 1.5, 't_fuze': 3.6}]
        }
    }
    plot_scenario(
        missile_ids=['M1'],
        strategies=strategy_q1_example,
        title="问题一：单机单弹最优策略示意图 (Y/Z轴已拉伸)",
        aspect_ratio_factors=(1, 8, 8)
    )
    plot_interception_details(
        missile_id='M1',
        drone_id='FY1',
        strategy=strategy_q1_example['FY1'],
        title="问题一：拦截细节放大图"
    )
    # 问题2绘图
    strategy_q2_example = {
        'FY1': {
            'params': {'theta': np.deg2rad(179.65), 'v_uav': 139.95},
            'grenades': [{'t_drop': 0.0223, 't_fuze': 3.6212}]
        }
    }
    plot_scenario(
        missile_ids=['M1'],
        strategies=strategy_q2_example,
        title="问题二：单机单弹最优策略示意图 (Y/Z轴已拉伸)",
        aspect_ratio_factors=(1, 8, 8)
    )
    plot_interception_details(
        missile_id='M1',
        drone_id='FY1',
        strategy=strategy_q2_example['FY1'],
        title="问题二：拦截细节放大图"
    )
    # 问题3绘图
    strategy_q3_example = {
        'FY1': {
            'params': {'theta': np.deg2rad(179.73), 'v_uav': 70.5},
            'grenades': [
                {'t_drop': 0.0, 't_fuze': 3.0212},
                {'t_drop': 3.2589, 't_fuze': 3.6636},
                {'t_drop': 5.4596, 't_fuze': 4.000}
            ]
        }
    }
    plot_scenario(
        missile_ids=['M1'],
        strategies=strategy_q3_example,
        title="问题三：单机单弹最优策略示意图 (Y/Z轴已拉伸)",
        aspect_ratio_factors=(1, 8, 8)
    )
    plot_interception_details(
        missile_id='M1',
        drone_id='FY1',
        strategy=strategy_q3_example['FY1'],
        title="问题三：拦截细节放大图"
    )
    #问题4绘图
    strategy_q4_example = {
        'FY1': {
            'params': {'theta': np.deg2rad(179.6541), 'v_uav': 139.9541},
            'grenades': [
                {'t_drop': 0.0123, 't_fuze': 3.6212}
            ]
        },
        'FY2': {
            'params': {'theta': np.deg2rad(307.7133), 'v_uav': 139.9749},
            'grenades': [
                {'t_drop': 8.1070, 't_fuze': 4.2301}
            ]
        },
        'FY3': {
            'params': {'theta': np.deg2rad(74.2896), 'v_uav': 136.3834},
            'grenades': [
                {'t_drop': 22.4056, 't_fuze': 1.1301}
            ]
        }
    }
    plot_scenario(
        missile_ids=['M1'],
        strategies=strategy_q4_example,
        title="问题四：多机单弹最优策略示意图 (Y/Z轴已拉伸)",
        aspect_ratio_factors=(1, 8, 8)
    )
    # plot_interception_details(
    #     missile_id='M1',
    #     drone_id=['FY1','FY2','FY3'],
    #     strategy=strategy_q4_example,
    #     title="问题四：拦截细节放大图"
    # )
    # 问题5绘图
    # strategy_q5_example = {
    #     'FY1': {
    #         'params': {'theta': np.deg2rad(179.50), 'v_uav': 139.52},
    #         'grenades': [
    #             {'t_drop': 0.0123, 't_fuze': 3.6212}
    #         ]
    #     }
    # }
