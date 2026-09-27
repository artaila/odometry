import numpy as np
from .ukf import UnscentedKalmanFilter

class TramDynamics:
    """Нелинейная модель продольной динамики трамвая."""
    def __init__(
        self,
        mass: float = 25000.0,
        k_traction: float = 1620.5,
        k_brake: float = 2200.0,
        davis_c0: float = 285.2,
        davis_c1: float = 18.4,
        davis_c2: float = 0.76,
        max_traction_force: float = 45000.0,
        max_brake_force: float = 55000.0,
        min_command: int = -15,
        max_command: int = 15,
        max_grade: float = 0.12,
        static_velocity_eps: float = 0.02,
        max_substep: float = 0.02,
    ):
        self.mass = float(mass)
        self.k_traction = float(k_traction)
        self.k_brake = float(k_brake)
        self.c0 = float(davis_c0)
        self.c1 = float(davis_c1)
        self.c2 = float(davis_c2)
        self.max_traction_force = float(max_traction_force)
        self.max_brake_force = float(max_brake_force)
        self.min_command = int(min_command)
        self.max_command = int(max_command)
        self.max_grade = float(max_grade)
        self.eps = float(static_velocity_eps)
        self.max_substep = float(max_substep)
        self.g = 9.81

    def traction_force(self, u: float, v: float = 0.0) -> float:
        u = float(np.clip(u, self.min_command, self.max_command))
        if u >= 0.0:
            force = self.k_traction * u
            return min(force, self.max_traction_force)
        force = self.k_brake * u
        return max(force, -self.max_brake_force)

    def resistance_force(self, v: float) -> float:
        v = max(0.0, float(v))
        return self.c0 + self.c1 * v + self.c2 * v * v

    def acceleration(self, v: float, u: float, grade: float = 0.0) -> float:
        v = max(0.0, float(v))
        grade = float(np.clip(grade, -self.max_grade, self.max_grade))

        f_tr = self.traction_force(u, v)
        f_res = self.resistance_force(v)
        f_grade = self.mass * self.g * grade
        net_force = f_tr - f_res - f_grade

        if v <= self.eps:
            if u <= 0.0 or net_force <= 0.0:
                return 0.0

        a = net_force / self.mass
        if v <= self.eps and a < 0.0:
            return 0.0
        return float(a)

    def step(self, s: float, v: float, u: float, dt: float, grade: float = 0.0):
        v = max(0.0, float(v))
        a = self.acceleration(v, u, grade)
        v_next = v + a * dt
        if v_next < 0.0:
            v_next = 0.0
        s_next = s + 0.5 * (v + v_next) * dt
        return s_next, v_next

    def integrate(self, s: float, v: float, u: float, dt: float, grade: float = 0.0):
        if dt <= 0.0 or not np.isfinite(dt):
            return max(0.0, float(s)), max(0.0, float(v))
        s = float(s)
        v = max(0.0, float(v))
        remaining = float(dt)
        while remaining > 1.0e-12:
            h = min(remaining, self.max_substep)
            s, v = self.step(s, v, u, h, grade)
            remaining -= h
        return s, v


class SlipDetector:
    """Адаптивный детектор проскальзывания и юза колесных пар."""
    def __init__(
        self,
        base_sigma: float = 0.05,
        max_inflation: float = 100.0,
        slip_min_inflation: float = 10.0,
        skip_inflation: float = 5.0,
        accel_abs_threshold: float = 1.5,
        accel_mismatch_threshold: float = 0.8,
        axle_diff_threshold: float = 0.35,
        innovation_base_threshold: float = 0.35,
        innovation_vel_threshold: float = 0.03,
        clear_cycles: int = 12,
        derivative_alpha: float = 0.35,
        use_abs_accel_threshold: bool = True,
    ):
        self.base_sigma = float(base_sigma)
        self.max_inflation = float(max_inflation)
        self.slip_min_inflation = float(slip_min_inflation)
        self.skip_inflation = float(skip_inflation)
        self.accel_abs_threshold = float(accel_abs_threshold)
        self.accel_mismatch_threshold = float(accel_mismatch_threshold)
        self.axle_diff_threshold = float(axle_diff_threshold)
        self.innovation_base_threshold = float(innovation_base_threshold)
        self.innovation_vel_threshold = float(innovation_vel_threshold)
        self.clear_cycles = int(clear_cycles)
        self.derivative_alpha = float(derivative_alpha)
        self.use_abs_accel_threshold = bool(use_abs_accel_threshold)

        self.inflation = 1.0
        self.clear_counter = 0
        self.prev_t = None
        self.prev_front = 0.0
        self.prev_rear = 0.0
        self.a_front = 0.0
        self.a_rear = 0.0

    def update(
        self,
        t: float,
        z_front: float,
        z_rear: float,
        v_pred: float,
        a_model: float,
        u: float,
    ):
        z_front = max(0.0, float(z_front))
        z_rear = max(0.0, float(z_rear))
        v_pred = max(0.0, float(v_pred))
        a_avg = 0.0

        if self.prev_t is not None:
            dt = float(t) - float(self.prev_t)
            if dt > 1.0e-6:
                a_front_raw = float(np.clip((z_front - self.prev_front) / dt, -20.0, 20.0))
                a_rear_raw = float(np.clip((z_rear - self.prev_rear) / dt, -20.0, 20.0))
                alpha = self.derivative_alpha
                self.a_front = alpha * a_front_raw + (1.0 - alpha) * self.a_front
                self.a_rear = alpha * a_rear_raw + (1.0 - alpha) * self.a_rear
                a_avg = 0.5 * (self.a_front + self.a_rear)

        self.prev_t = float(t)
        self.prev_front = z_front
        self.prev_rear = z_rear

        innov_front = abs(z_front - v_pred)
        innov_rear = abs(z_rear - v_pred)
        innov_avg = 0.5 * (innov_front + innov_rear)
        axle_diff = abs(z_front - z_rear)

        slip = False
        reasons = []
        innov_thr = self.innovation_base_threshold + self.innovation_vel_threshold * abs(v_pred)

        if innov_avg > innov_thr:
            slip = True
            reasons.append("innovation")
        if axle_diff > self.axle_diff_threshold:
            slip = True
            reasons.append("axle_diff")
        if abs(a_avg - a_model) > self.accel_mismatch_threshold:
            slip = True
            reasons.append("accel_mismatch")
        if self.use_abs_accel_threshold and abs(a_avg) > self.accel_abs_threshold and abs(a_model) < 0.75 * self.accel_abs_threshold:
            slip = True
            reasons.append("abs_accel")

        if slip:
            self.clear_counter = 0
            self.inflation = min(self.max_inflation, max(self.inflation * 2.0, self.slip_min_inflation))
        else:
            self.clear_counter += 1
            if self.clear_counter >= self.clear_cycles:
                self.inflation = max(1.0, self.inflation * 0.8)

        sigma = self.base_sigma * self.inflation
        R = np.eye(2, dtype=float) * (sigma * sigma)
        use_measurement = self.inflation < self.skip_inflation

        info = {
            "slip": slip,
            "reasons": reasons,
            "inflation": self.inflation,
            "a_wheel": a_avg,
            "a_model": a_model,
            "innov_avg": innov_avg,
            "axle_diff": axle_diff,
        }
        return R, use_measurement, info


class DynamicsEstimator:
    """Оценщик состояния трамвая на основе UKF и динамической модели."""
    def __init__(
        self,
        dynamics_params: dict = None,
        estimate_grade: bool = False,
        static_grade: float = 0.0,
        max_grade: float = 0.12,
        q_s: float = 0.1,
        q_v: float = 0.5,
        q_grade: float = 0.001,
        ukf_alpha: float = 1.0,
        ukf_beta: float = 2.0,
        ukf_kappa: float = 0.0,
    ):
        dynamics_params = dynamics_params or {}
        if "max_grade" not in dynamics_params:
            dynamics_params["max_grade"] = max_grade

        self.dyn = TramDynamics(**dynamics_params)
        self.estimate_grade = bool(estimate_grade)
        self.static_grade = float(static_grade)
        self.max_grade = float(max_grade)
        self.q_s = float(q_s)
        self.q_v = float(q_v)
        self.q_grade = float(q_grade)

        dim_x = 3 if self.estimate_grade else 2
        dim_z = 2
        self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, alpha=ukf_alpha, beta=ukf_beta, kappa=ukf_kappa)
        self.initialized = False
        self.ukf.P = np.diag([25.0, 1.0] + ([1.0e-4] if self.estimate_grade else []))

    def initialize(self, s0: float = 0.0, v0: float = 0.0, grade0: float = None):
        if grade0 is None:
            grade0 = self.static_grade
        v0 = max(0.0, float(v0))
        s0 = max(0.0, float(s0))
        if self.estimate_grade:
            self.ukf.x = np.array([s0, v0, float(grade0)], dtype=float)
        else:
            self.ukf.x = np.array([s0, v0], dtype=float)
        self._clip_state()
        self.initialized = True

    def _f(self, x, u, dt):
        s = float(x[0])
        v = max(0.0, float(x[1]))
        grade = float(np.clip(x[2], -self.max_grade, self.max_grade)) if self.estimate_grade else self.static_grade
        s2, v2 = self.dyn.integrate(s, v, u, dt, grade)
        if self.estimate_grade:
            return np.array([s2, v2, grade], dtype=float)
        return np.array([s2, v2], dtype=float)

    def _h(self, x):
        v = max(0.0, float(x[1]))
        return np.array([v, v], dtype=float)

    def _process_noise(self, dt: float):
        if dt <= 0.0:
            return np.zeros((self.ukf.n, self.ukf.n), dtype=float)
        q_diag = [(self.q_s * dt) ** 2, (self.q_v * dt) ** 2]
        if self.estimate_grade:
            q_diag.append((self.q_grade * dt) ** 2)
        return np.diag(q_diag)

    def predict(self, dt: float, u: float):
        if not self.initialized or dt <= 0.0:
            return
        Q = self._process_noise(dt)
        self.ukf.predict(self._f, dt, u, Q)
        self._clip_state()

    def update_wheel_measurements(self, z_front: float, z_rear: float, R):
        if not self.initialized:
            return
        z = np.array([max(0.0, float(z_front)), max(0.0, float(z_rear))], dtype=float)
        self.ukf.update(z, R, self._h)
        self._clip_state()

    def _clip_state(self):
        self.ukf.x[0] = max(0.0, float(self.ukf.x[0]))
        self.ukf.x[1] = max(0.0, float(self.ukf.x[1]))
        if self.estimate_grade:
            self.ukf.x[2] = float(np.clip(self.ukf.x[2], -self.max_grade, self.max_grade))

    @property
    def s(self) -> float:
        return float(self.ukf.x[0])

    @property
    def v(self) -> float:
        return float(self.ukf.x[1])

    @property
    def grade(self) -> float:
        return float(self.ukf.x[2]) if self.estimate_grade else self.static_grade

    def model_acceleration(self, u: float) -> float:
        return self.dyn.acceleration(self.ukf.x[1], u, self.grade)

    def zero_velocity(self):
        self.ukf.x[1] = 0.0
        if self.ukf.P.shape[0] > 1:
            self.ukf.P[1, 1] *= 0.5
