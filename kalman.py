"""
EKF 与 UKF 对比教学示例
=======================

场景：二维平面中的机动目标。

状态向量为
    x = [px, py, v, heading, turn_rate]^T

其中 (px, py) 是位置，v 是速度，heading 是航向角，turn_rate 是转弯角速度。
运动模型含有 sin/cos，因此是非线性的；传感器输出距离和方位角，观测模型
含有 sqrt/atan2，同样是强非线性的。

运行：
    python kalman.py

依赖：numpy、matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt


# Matplotlib 默认字体通常不包含中文，缺字时会显示成方框。
# 这里从当前系统已安装字体中选择第一个可用的中文字体。
def configure_chinese_font() -> None:
    preferred_fonts = [
        "Microsoft YaHei",  # Windows 常见：微软雅黑
        "SimHei",           # Windows 常见：黑体
        "SimSun",           # Windows 常见：宋体
        "Noto Sans CJK SC", # Linux 常见：思源/ Noto 中文字体
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((name for name in preferred_fonts
                          if name in installed_fonts), None)
    if selected_font is not None:
        # 把所选字体放在候选列表第一位，图例、标题、坐标轴标签都会继承它。
        matplotlib.rcParams["font.sans-serif"] = [selected_font] + preferred_fonts
    else:
        print("提示：未找到常见中文字体，图中的中文可能显示为方框。")
    # 避免负号被渲染成方框或乱码。
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_chinese_font()


# ----------------------------- 公共工具 -----------------------------

STATE_SIZE = 5
ANGLE_INDEX = 3  # 状态中的 heading 是角度，需要特殊处理均值和残差


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """把角度包装到 [-pi, pi)。"""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def angle_residual(a: np.ndarray, b: np.ndarray, index: int) -> np.ndarray:
    """计算向量残差，并对指定角度分量做环绕处理。"""
    residual = a - b
    residual[..., index] = wrap_angle(residual[..., index])
    return residual


def weighted_angle_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """用圆周均值计算角度的加权平均，避免 pi/-pi 跳变。"""
    return float(np.arctan2(np.sum(weights * np.sin(values)),
                            np.sum(weights * np.cos(values))))


# ----------------------------- 非线性系统 -----------------------------


def process_model(state: np.ndarray, dt: float, t: float) -> np.ndarray:
    """
    非线性运动模型 f(x_k, t)：

        px_{k+1} = px_k + v_k cos(theta_k) dt
        py_{k+1} = py_k + v_k sin(theta_k) dt
        v_{k+1}  = v_k + (-drag*v_k + a_drive(t)) dt
        theta_{k+1} = theta_k + omega_k dt
        omega_{k+1} = omega_k

    驱动力随时间变化，目标会加减速并持续转弯。这里不把过程噪声加到
    函数内部，噪声在仿真和滤波器的协方差 Q 中分别体现。
    """
    px, py, speed, heading, turn_rate = state
    drag = 0.08
    drive = 0.35 * np.sin(0.18 * t) + 0.12 * np.cos(0.47 * t)

    next_state = np.array([
        px + speed * np.cos(heading) * dt,
        py + speed * np.sin(heading) * dt,
        speed + (-drag * speed + drive) * dt,
        wrap_angle(heading + turn_rate * dt),
        turn_rate,
    ])
    return next_state


def measurement_model(state: np.ndarray) -> np.ndarray:
    """
    距离-方位角观测模型 h(x)：

        range    = sqrt(px^2 + py^2)
        bearing  = atan2(py, px)

    传感器位于原点，观测不直接给出 x/y 坐标，所以当目标距离较近或
    方位角变化较快时，局部线性化误差会明显增大。
    """
    px, py = state[:2]
    distance = np.hypot(px, py)
    bearing = np.arctan2(py, px)
    return np.array([distance, bearing])


def process_jacobian(state: np.ndarray, dt: float, t: float) -> np.ndarray:
    """
    运动模型的解析雅可比 F = df/dx。

    例如：
        d(px_next)/d(v)       = cos(theta) dt
        d(px_next)/d(theta)   = -v sin(theta) dt
        d(theta_next)/d(omega)= dt

    EKF 在每一步都用当前估计点处的 F 近似非线性函数。
    """
    _, _, speed, heading, _ = state
    drag = 0.08
    return np.array([
        [1.0, 0.0, np.cos(heading) * dt, -speed * np.sin(heading) * dt, 0.0],
        [0.0, 1.0, np.sin(heading) * dt, speed * np.cos(heading) * dt, 0.0],
        [0.0, 0.0, 1.0 - drag * dt, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, dt],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])


def measurement_jacobian(state: np.ndarray) -> np.ndarray:
    """
    观测模型的解析雅可比 H = dh/dx。

    令 r = sqrt(px^2 + py^2)，则
        d(range)/d(px,py) = (px/r, py/r)
        d(bearing)/d(px,py) = (-py/r^2, px/r^2)

    观测对速度、航向和转弯角速度没有直接偏导，因此对应列为 0。
    """
    px, py = state[:2]
    r2 = max(px * px + py * py, 1e-12)
    r = np.sqrt(r2)  # 防止目标恰好位于传感器原点
    return np.array([
        [px / r, py / r, 0.0, 0.0, 0.0],
        [-py / r2, px / r2, 0.0, 0.0, 0.0],
    ])


# ----------------------------- EKF -----------------------------


def ekf_filter(zs: np.ndarray, initial_state: np.ndarray, initial_cov: np.ndarray,
               dt: float, times: np.ndarray, Q: np.ndarray,
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """扩展卡尔曼滤波：预测和更新均在当前估计点做一阶泰勒展开。"""
    estimates = np.zeros((len(zs), STATE_SIZE))
    covariances = np.zeros((len(zs), STATE_SIZE, STATE_SIZE))
    state = initial_state.copy()
    covariance = initial_cov.copy()

    for k, measurement in enumerate(zs):
        # ---------- 预测步骤 ----------
        # x^-_k = f(x^+_{k-1}, t_k)
        predicted_state = process_model(state, dt, times[k - 1] if k else 0.0)
        # P^-_k = F_k P^+_{k-1} F_k^T + Q
        F = process_jacobian(state, dt, times[k - 1] if k else 0.0)
        predicted_covariance = F @ covariance @ F.T + Q

        # ---------- 更新步骤 ----------
        # y_k = z_k - h(x^-_k)，其中方位角残差必须进行角度环绕。
        H = measurement_jacobian(predicted_state)
        innovation = measurement - measurement_model(predicted_state)
        innovation[1] = wrap_angle(innovation[1])
        # S_k = H P^-_k H^T + R：创新（观测预测误差）的协方差。
        innovation_covariance = H @ predicted_covariance @ H.T + R
        # K_k = P^-_k H^T S_k^{-1}：卡尔曼增益。
        gain = predicted_covariance @ H.T @ np.linalg.inv(innovation_covariance)
        # x^+_k = x^-_k + K_k y_k
        state = predicted_state + gain @ innovation
        state[ANGLE_INDEX] = wrap_angle(state[ANGLE_INDEX])
        # Joseph 形式：P^+ = (I-KH)P^-(I-KH)^T + K R K^T，数值更稳定。
        identity = np.eye(STATE_SIZE)
        covariance = ((identity - gain @ H) @ predicted_covariance
                      @ (identity - gain @ H).T + gain @ R @ gain.T)

        estimates[k] = state
        covariances[k] = covariance

    return estimates, covariances


# ----------------------------- UKF -----------------------------


def sigma_points(state: np.ndarray, covariance: np.ndarray,
                 alpha: float, beta: float, kappa: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 2n+1 个 sigma 点。

    lambda = alpha^2(n+kappa)-n
    chi_0 = x
    chi_i = x +/- sqrt((n+lambda)P)_i

    sqrt 使用 Cholesky 分解，保证 sigma 点沿协方差主轴展开。
    """
    n = len(state)
    lam = alpha * alpha * (n + kappa) - n
    scale = n + lam
    # 对称化后，将极小的负特征值截断为正数，避免浮点误差或强非线性
    # 更新造成的轻微非正定，使 Cholesky 分解能够稳定生成 Sigma 点。
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    jitter = 1e-9 * np.eye(n)
    square_root = np.linalg.cholesky(scale * covariance + jitter)

    points = np.zeros((2 * n + 1, n))
    points[0] = state
    points[1:n + 1] = state + square_root.T
    points[n + 1:] = state - square_root.T
    points[:, ANGLE_INDEX] = wrap_angle(points[:, ANGLE_INDEX])

    weights_mean = np.full(2 * n + 1, 1.0 / (2.0 * scale))
    weights_covariance = weights_mean.copy()
    weights_mean[0] = lam / scale
    weights_covariance[0] = lam / scale + (1.0 - alpha * alpha + beta)
    return points, weights_mean, weights_covariance


def state_weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """sigma 点状态均值，heading 分量使用圆周均值。"""
    mean = np.sum(weights[:, None] * points, axis=0)
    mean[ANGLE_INDEX] = weighted_angle_mean(points[:, ANGLE_INDEX], weights)
    return mean


def ukf_filter(zs: np.ndarray, initial_state: np.ndarray, initial_cov: np.ndarray,
               dt: float, times: np.ndarray, Q: np.ndarray,
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """无迹卡尔曼滤波：用 sigma 点的非线性传播近似均值和协方差。"""
    estimates = np.zeros((len(zs), STATE_SIZE))
    covariances = np.zeros((len(zs), STATE_SIZE, STATE_SIZE))
    state = initial_state.copy()
    covariance = initial_cov.copy()
    # 初始协方差较大时，适当扩大 sigma 点分布，避免只在均值附近取样。
    alpha, beta, kappa = 0.50, 2.0, 0.0

    for k, measurement in enumerate(zs):
        # ---------- 预测步骤 ----------
        points, wm, wc = sigma_points(state, covariance, alpha, beta, kappa)
        propagated_points = np.array([
            process_model(point, dt, times[k - 1] if k else 0.0)
            for point in points
        ])
        # x^- = sum_i Wm_i f(chi_i)
        predicted_state = state_weighted_mean(propagated_points, wm)
        # P^- = sum_i Wc_i [f(chi_i)-x^-][...]^T + Q
        predicted_covariance = Q.copy()
        for i in range(len(propagated_points)):
            diff = angle_residual(propagated_points[i], predicted_state, ANGLE_INDEX)
            predicted_covariance += wc[i] * np.outer(diff, diff)

        # ---------- 观测变换 ----------
        # 将预测 sigma 点通过 h(.)，得到观测空间的 sigma 点。
        measurement_points = np.array([measurement_model(point)
                                       for point in propagated_points])
        predicted_measurement = np.sum(wm[:, None] * measurement_points, axis=0)
        # 方位角是圆变量，不能直接用普通加权平均。
        predicted_measurement[1] = weighted_angle_mean(
            measurement_points[:, 1], wm)

        # S = sum_i Wc_i dz_i dz_i^T + R
        innovation_covariance = R.copy()
        # C = sum_i Wc_i dx_i dz_i^T，状态和观测之间的互协方差。
        cross_covariance = np.zeros((STATE_SIZE, 2))
        for i in range(len(measurement_points)):
            state_diff = angle_residual(propagated_points[i], predicted_state,
                                        ANGLE_INDEX)
            measurement_diff = measurement_points[i] - predicted_measurement
            measurement_diff[1] = wrap_angle(measurement_diff[1])
            innovation_covariance += wc[i] * np.outer(measurement_diff,
                                                      measurement_diff)
            cross_covariance += wc[i] * np.outer(state_diff, measurement_diff)

        # K = C S^{-1}，避免显式求逆时也可以改用 np.linalg.solve。
        gain = cross_covariance @ np.linalg.inv(innovation_covariance)
        innovation = measurement - predicted_measurement
        innovation[1] = wrap_angle(innovation[1])
        # x^+ = x^- + K(z-z^-)，P^+ = P^- - K S K^T
        state = predicted_state + gain @ innovation
        state[ANGLE_INDEX] = wrap_angle(state[ANGLE_INDEX])
        covariance = predicted_covariance - gain @ innovation_covariance @ gain.T
        covariance = 0.5 * (covariance + covariance.T)

        estimates[k] = state
        covariances[k] = covariance

    return estimates, covariances


# ----------------------------- 仿真与绘图 -----------------------------


def simulate(steps: int = 260, dt: float = 0.30,
             seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成真实轨迹和距离/方位角观测。"""
    rng = np.random.default_rng(seed)
    times = np.arange(steps) * dt
    true_states = np.zeros((steps, STATE_SIZE))
    measurements = np.zeros((steps, 2))
    # 让目标从传感器附近出发，放大 atan2 和距离模型的局部非线性。
    true_states[0] = np.array([0.6, 0.4, 1.35, 0.65, 0.30])

    # 过程噪声标准差：位置、速度、航向、转弯速率分别对应不同量纲。
    process_std = np.array([0.15, 0.15, 0.09, 0.010, 0.04])
    Q_sim = np.diag(process_std ** 2)
    # 观测噪声：距离约 0.30 m，方位约 8.0 度。
    R = np.diag([0.30 ** 2, np.deg2rad(8.0) ** 2])

    measurements[0] = measurement_model(true_states[0]) + rng.multivariate_normal(
        np.zeros(2), R)
    measurements[0, 1] = wrap_angle(measurements[0, 1])

    for k in range(1, steps):
        # x_k = f(x_{k-1}, t_{k-1}) + q_k
        true_states[k] = (process_model(true_states[k - 1], dt, times[k - 1])
                          + rng.multivariate_normal(np.zeros(STATE_SIZE), Q_sim))
        true_states[k, ANGLE_INDEX] = wrap_angle(true_states[k, ANGLE_INDEX])
        measurements[k] = (measurement_model(true_states[k])
                           + rng.multivariate_normal(np.zeros(2), R))
        measurements[k, 1] = wrap_angle(measurements[k, 1])

    return times, true_states, measurements


def plot_results(times: np.ndarray, true_states: np.ndarray,
                 measurements: np.ndarray, ekf_states: np.ndarray,
                 ukf_states: np.ndarray) -> None:
    """绘制轨迹、位置误差、速度估计和方位角估计。"""
    measured_xy = np.column_stack([
        measurements[:, 0] * np.cos(measurements[:, 1]),
        measurements[:, 0] * np.sin(measurements[:, 1]),
    ])
    true_position = true_states[:, :2]
    ekf_position = ekf_states[:, :2]
    ukf_position = ukf_states[:, :2]
    ekf_error = np.linalg.norm(ekf_position - true_position, axis=1)
    ukf_error = np.linalg.norm(ukf_position - true_position, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # 1) 平面轨迹：极坐标观测先转换回 x/y，仅用于可视化。
    ax = axes[0, 0]
    ax.plot(true_position[:, 0], true_position[:, 1], "k-", lw=2.2,
            label="真实轨迹")
    ax.scatter(measured_xy[:, 0], measured_xy[:, 1], s=10, alpha=0.25,
               color="gray", label="距离/方位观测")
    ax.plot(ekf_position[:, 0], ekf_position[:, 1], "tab:blue", lw=1.5,
            label="EKF")
    ax.plot(ukf_position[:, 0], ukf_position[:, 1], "tab:orange", lw=1.5,
            label="UKF")
    ax.scatter([0], [0], marker="^", s=100, color="tab:red", label="传感器")
    ax.set_title("二维轨迹对比")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()

    # 2) 位置误差：UKF 通常在强非线性区域具有更小的线性化误差。
    ax = axes[0, 1]
    ax.plot(times, ekf_error, color="tab:blue", label="EKF 位置误差")
    ax.plot(times, ukf_error, color="tab:orange", label="UKF 位置误差")
    ax.set_title("位置误差范数")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("||位置误差|| / m")
    ax.grid(alpha=0.25)
    ax.legend()

    # 3) 速度估计。
    ax = axes[1, 0]
    ax.plot(times, true_states[:, 2], "k-", lw=2, label="真实速度")
    ax.plot(times, ekf_states[:, 2], color="tab:blue", label="EKF")
    ax.plot(times, ukf_states[:, 2], color="tab:orange", label="UKF")
    ax.set_title("速度估计")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("速度 / (m/s)")
    ax.grid(alpha=0.25)
    ax.legend()

    # 4) 航向角估计，绘图前解包裹，使曲线连续。
    ax = axes[1, 1]
    ax.plot(times, np.unwrap(true_states[:, 3]), "k-", lw=2, label="真实航向")
    ax.plot(times, np.unwrap(ekf_states[:, 3]), color="tab:blue", label="EKF")
    ax.plot(times, np.unwrap(ukf_states[:, 3]), color="tab:orange", label="UKF")
    ax.set_title("航向角估计")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("航向角 / rad")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("强非线性机动目标：EKF 与 UKF 对比", fontsize=15)
    plt.show()


def main() -> None:
    times, true_states, measurements = simulate()
    dt = times[1] - times[0]

    # 故意给滤波器一个明显有偏的初始状态，观察强非线性下的收敛过程。
    initial_state = np.array([0.5, 0.5, 2.0, -1.5, 0.8])
    initial_covariance = np.diag([
        2.0 ** 2, 2.0 ** 2, 1.0 ** 2,
        np.deg2rad(45.0) ** 2, np.deg2rad(20.0) ** 2,
    ])

    # 滤波器认为的过程噪声和传感器噪声（可以自行调大/调小观察效果）。
    # 与 simulate() 中真实过程噪声保持一致，保证 EKF/UKF 比较公平。
    Q = np.diag([0.15 ** 2, 0.15 ** 2, 0.09 ** 2,
                 0.010 ** 2, 0.04 ** 2])
    R = np.diag([0.30 ** 2, np.deg2rad(8.0) ** 2])

    ekf_states, _ = ekf_filter(measurements, initial_state, initial_covariance,
                               dt, times, Q, R)
    ukf_states, _ = ukf_filter(measurements, initial_state, initial_covariance,
                               dt, times, Q, R)

    ekf_rmse = np.sqrt(np.mean(np.sum((ekf_states[:, :2] - true_states[:, :2]) ** 2,
                                      axis=1)))
    ukf_rmse = np.sqrt(np.mean(np.sum((ukf_states[:, :2] - true_states[:, :2]) ** 2,
                                      axis=1)))
    print(f"EKF 平均位置 RMSE: {ekf_rmse:.4f} m")
    print(f"UKF 平均位置 RMSE: {ukf_rmse:.4f} m")

    plot_results(times, true_states, measurements, ekf_states, ukf_states)


if __name__ == "__main__":
    main()
