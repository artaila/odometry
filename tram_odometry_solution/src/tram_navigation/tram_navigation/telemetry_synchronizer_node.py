import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from .dynamics_estimator import DynamicsEstimator, SlipDetector
from .sync_buffer import SyncBuffer
from .track_geometry import TrackGeometry

class TelemetrySynchronizerNode(Node):
    """ROS 2 узел резервной одометрии беспилотного трамвая."""
    def __init__(self):
        super().__init__("tram_backup_odometry")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("sync_window_sec", 0.05),
                ("max_buffer_age_sec", 0.5),
                ("min_publish_period_sec", 0.045),
                ("max_dt_sec", 0.25),
                ("min_velocity", 0.0),
                ("max_velocity", 35.0),
                ("min_command", -15),
                ("max_command", 15),
                ("mass", 25000.0),
                ("k_traction", 1620.5),
                ("k_brake", 2200.0),
                ("davis_c0", 285.2),
                ("davis_c1", 18.4),
                ("davis_c2", 0.76),
                ("max_traction_force", 45000.0),
                ("max_brake_force", 55000.0),
                ("max_grade", 0.12),
                ("estimate_grade", False),
                ("initial_grade", 0.0),
                ("q_s", 0.10),
                ("q_v", 0.50),
                ("q_grade", 0.001),
                ("base_wheel_sigma", 0.05),
                ("slip_inflation_max", 100.0),
                ("slip_min_inflation", 10.0),
                ("slip_skip_inflation", 5.0),
                ("slip_accel_abs_threshold", 1.5),
                ("slip_accel_mismatch_threshold", 0.8),
                ("slip_diff_threshold", 0.35),
                ("slip_innovation_threshold", 0.35),
                ("slip_innovation_vel_threshold", 0.03),
                ("slip_clear_cycles", 12),
                ("zero_velocity_threshold", 0.03),
                ("odom_frame", "odom"),
                ("base_frame", "base_link"),
            ],
        )

        p = self.get_parameter

        self.sync_window = float(p("sync_window_sec").value)
        self.max_buffer_age = float(p("max_buffer_age_sec").value)
        self.min_publish_period = float(p("min_publish_period_sec").value)
        self.max_dt = float(p("max_dt_sec").value)

        self.min_velocity = float(p("min_velocity").value)
        self.max_velocity = float(p("max_velocity").value)
        self.min_command = int(p("min_command").value)
        self.max_command = int(p("max_command").value)

        self.mass = float(p("mass").value)
        self.k_traction = float(p("k_traction").value)
        self.k_brake = float(p("k_brake").value)
        self.davis_c0 = float(p("davis_c0").value)
        self.davis_c1 = float(p("davis_c1").value)
        self.davis_c2 = float(p("davis_c2").value)
        self.max_traction_force = float(p("max_traction_force").value)
        self.max_brake_force = float(p("max_brake_force").value)
        self.max_grade = float(p("max_grade").value)

        self.estimate_grade = bool(p("estimate_grade").value)
        self.initial_grade = float(p("initial_grade").value)

        self.q_s = float(p("q_s").value)
        self.q_v = float(p("q_v").value)
        self.q_grade = float(p("q_grade").value)

        self.base_wheel_sigma = float(p("base_wheel_sigma").value)
        self.slip_inflation_max = float(p("slip_inflation_max").value)
        self.slip_min_inflation = float(p("slip_min_inflation").value)
        self.slip_skip_inflation = float(p("slip_skip_inflation").value)
        self.slip_accel_abs_threshold = float(p("slip_accel_abs_threshold").value)
        self.slip_accel_mismatch_threshold = float(p("slip_accel_mismatch_threshold").value)
        self.slip_diff_threshold = float(p("slip_diff_threshold").value)
        self.slip_innovation_threshold = float(p("slip_innovation_threshold").value)
        self.slip_innovation_vel_threshold = float(p("slip_innovation_vel_threshold").value)
        self.slip_clear_cycles = int(p("slip_clear_cycles").value)

        self.zero_velocity_threshold = float(p("zero_velocity_threshold").value)
        self.odom_frame = str(p("odom_frame").value)
        self.base_frame = str(p("base_frame").value)

        self.sync = SyncBuffer(
            topics=["front", "rear", "cmd"],
            window_sec=self.sync_window,
            max_age_sec=self.max_buffer_age,
            min_interval_sec=self.min_publish_period,
        )

        dynamics_params = {
            "mass": self.mass,
            "k_traction": self.k_traction,
            "k_brake": self.k_brake,
            "davis_c0": self.davis_c0,
            "davis_c1": self.davis_c1,
            "davis_c2": self.davis_c2,
            "max_traction_force": self.max_traction_force,
            "max_brake_force": self.max_brake_force,
            "min_command": self.min_command,
            "max_command": self.max_command,
            "max_grade": self.max_grade,
        }

        self.estimator = DynamicsEstimator(
            dynamics_params=dynamics_params,
            estimate_grade=self.estimate_grade,
            static_grade=self.initial_grade,
            max_grade=self.max_grade,
            q_s=self.q_s,
            q_v=self.q_v,
            q_grade=self.q_grade,
        )

        self.slip = SlipDetector(
            base_sigma=self.base_wheel_sigma,
            max_inflation=self.slip_inflation_max,
            slip_min_inflation=self.slip_min_inflation,
            skip_inflation=self.slip_skip_inflation,
            accel_abs_threshold=self.slip_accel_abs_threshold,
            accel_mismatch_threshold=self.slip_accel_mismatch_threshold,
            axle_diff_threshold=self.slip_diff_threshold,
            innovation_base_threshold=self.slip_innovation_threshold,
            innovation_vel_threshold=self.slip_innovation_vel_threshold,
            clear_cycles=self.slip_clear_cycles,
        )

        self.track = TrackGeometry(points=None)

        self.pub_vel = self.create_publisher(VelocitySensor, "/result/velocity", 10)
        self.pub_odom = self.create_publisher(Odometry, "/result/position", 10)

        self.create_subscription(
            VelocitySensor,
            "/vehicle/front_bogie_velocity",
            self.front_cb,
            10,
        )
        self.create_subscription(
            VelocitySensor,
            "/vehicle/rear_bogie_velocity",
            self.rear_cb,
            10,
        )
        self.create_subscription(
            DriverControllerCommand,
            "/vehicle/driver_position_cmd",
            self.cmd_cb,
            10,
        )

        self.last_time = None

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    @staticmethod
    def _fill_stamp(stamp, t: float):
        sec = int(math.floor(t))
        nanosec = int(round((t - sec) * 1.0e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        stamp.sec = sec
        stamp.nanosec = nanosec

    def front_cb(self, msg: VelocitySensor):
        self._velocity_cb(msg, "front")

    def rear_cb(self, msg: VelocitySensor):
        self._velocity_cb(msg, "rear")

    def _velocity_cb(self, msg: VelocitySensor, topic: str):
        v = float(msg.velocity) / 3.6 # km/h to m/s
        if not math.isfinite(v):
            return
        if v < self.min_velocity or v > self.max_velocity:
            return
        t = self._stamp_to_sec(msg.header.stamp)
        self.sync.add(topic, t, v)
        self._try_process(topic)

    def cmd_cb(self, msg: DriverControllerCommand):
        u = int(msg.position)
        if u < self.min_command or u > self.max_command:
            return
        t = self._stamp_to_sec(msg.header.stamp)
        self.sync.add("cmd", t, float(u))
        self._try_process("cmd")

    def _try_process(self, trigger_topic: str):
        match = self.sync.latest_frame(trigger_topic)
        if match is None:
            return

        try:
            z_front = float(match.samples["front"].msg)
            z_rear = float(match.samples["rear"].msg)
            u = float(match.samples["cmd"].msg)
        except Exception:
            return

        self._process_frame(match.stamp, z_front, z_rear, u)

    def _process_frame(self, t: float, z_front: float, z_rear: float, u: float):
        if not self.estimator.initialized:
            v0 = 0.5 * (z_front + z_rear)
            v0 = min(max(v0, self.min_velocity), self.max_velocity)
            self.estimator.initialize(s0=0.0, v0=v0, grade0=self.initial_grade)
            self.last_time = t
            self._publish(t)
            return

        if t <= self.last_time:
            return

        dt = t - self.last_time
        dt = min(max(dt, 0.0), self.max_dt)
        if dt <= 1.0e-6:
            return

        self.estimator.predict(dt, u)
        v_pred = self.estimator.v
        a_model = self.estimator.model_acceleration(u)

        R, use_measurement, slip_info = self.slip.update(
            t=t,
            z_front=z_front,
            z_rear=z_rear,
            v_pred=v_pred,
            a_model=a_model,
            u=u,
        )

        if use_measurement:
            self.estimator.update_wheel_measurements(z_front, z_rear, R)

        avg_wheel = 0.5 * (z_front + z_rear)
        if abs(u) < 1.0e-6 and self.estimator.v < self.zero_velocity_threshold and avg_wheel < self.zero_velocity_threshold:
            self.estimator.zero_velocity()

        self._publish(t)
        self.last_time = t

    def _publish(self, t: float):
        s = self.estimator.s
        v = self.estimator.v

        vel_msg = VelocitySensor()
        self._fill_stamp(vel_msg.header.stamp, t)
        vel_msg.header.frame_id = self.base_frame
        vel_msg.velocity = float(v)
        self.pub_vel.publish(vel_msg)

        x, y, yaw = self.track.pose_at(s)
        odom = Odometry()
        self._fill_stamp(odom.header.stamp, t)
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = 0.0

        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0

        P = self.estimator.ukf.P
        odom.pose.covariance[0] = max(0.0, float(P[0, 0]))
        odom.pose.covariance[7] = 0.25
        odom.pose.covariance[14] = 0.01
        odom.pose.covariance[35] = 1.0

        if P.shape[0] > 1:
            odom.twist.covariance[0] = max(0.0, float(P[1, 1]))

        self.pub_odom.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = TelemetrySynchronizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
