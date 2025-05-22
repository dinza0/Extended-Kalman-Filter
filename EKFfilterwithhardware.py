import numpy as np
import ppigrf
import time
import serial
import aprslib
from datetime import datetime
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import socket
try:
    import board
    import busio
    import adafruit_bno08x
except (ImportError, NotImplementedError):
    board = busio = adafruit_bno08x = None
    # define minimal stubs so code that references them won’t crash:
    class DummySensor:
        quaternion = (1,0,0,0)
        acceleration = (0,0,0)
        gyro = (0,0,1)
        magnetic = (1,0,0)
        euler = None
    def init_bno085():
        return DummySensor()
    def read_bno085(sensor):
        return {
            "quaternion": sensor.quaternion,
            "acceleration": sensor.acceleration,
            "gyro": sensor.gyro,
            "magnetometer": sensor.magnetic,
            "euler": sensor.euler
        }

def quat_to_rotmat(q):
    """
    Convert a quaternion (vector-first [qx, qy, qz, qw]) to a rotation matrix.
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])
    return R

# --- Before your main EKF loop, set up an interactive 3D plot ---
plt.ion()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
ax.set_xlabel('I X'); ax.set_ylabel('I Y'); ax.set_zlabel('I Z')
ax.set_title('Body Axes in Inertial Frame')

def update_attitude_plot(q):
    """Clear and redraw body axes according to quaternion q."""
    R = quat_to_rotmat(q)
    origin = np.zeros(3)
    ax.cla()
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('I X'); ax.set_ylabel('I Y'); ax.set_zlabel('I Z')
    ax.set_title('Body Axes in Inertial Frame')
    # Draw body-frame axes
    ax.quiver(*origin, *R[:,0], length=1.0, normalize=True, color='r')  # Body X
    ax.quiver(*origin, *R[:,1], length=1.0, normalize=True, color='g')  # Body Y
    ax.quiver(*origin, *R[:,2], length=1.0, normalize=True, color='b')  # Body Z
    plt.draw()
    plt.pause(0.001)

def send_euler_angles(euler, target_ip="255.255.255.255", target_port=5005):
    """
    Broadcasts Euler angles over UDP.

    Parameters
    ----------
    euler : tuple of float
        (yaw, pitch, roll), in degrees (or radians) as you prefer.
    target_ip : str
        IP address to send to. Using the broadcast address (255.255.255.255)
        sends to all listeners on the local network.
    target_port : int
        UDP port number to send the data on.
    """
    # format the message as CSV
    # e.g. "30.0, -10.5, 5.2"
    msg = "{:.6f},{:.6f},{:.6f}".format(*euler)
    data = msg.encode("utf-8")

    # create UDP socket, enable broadcast
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    # send the packet
    sock.sendto(data, (target_ip, target_port))
    sock.close()


def read_aprs_gps(port="/dev/ttyUSB0", baudrate=9600, timeout=1):
    """
    Reads an APRS packet from a serial-connected APRS tracker and extracts GPS data.

    This function connects to the specified serial port and reads APRS data. The aprslib
    library is used to parse the packet and extract the latitude, longitude, and altitude.

    Parameters
    ----------
    port : str, optional
        The serial port to which the APRS tracker is connected (default: '/dev/ttyUSB0').
    baudrate : int, optional
        The baud rate for the serial connection (default: 9600).
    timeout : int or float, optional
        The serial port read timeout in seconds (default: 1).

    Returns
    -------
    lat : float
        Latitude in degrees (north positive), or None if not available.
    lon : float
        Longitude in degrees (east positive), or None if not available.
    alt : float
        Altitude in meters (if available), or None.
    """
    # Open the serial connection.
    ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    while True:
        try:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            # APRS packets usually start with a callsign or a '$' if NMEA sentences are output.
            # We attempt to parse the line as an APRS packet.
            packet = aprslib.parse(line)

            # Check for GPS data. aprslib returns a dict with keys such as 'latitude',
            # 'longitude', and (optionally) 'altitude' if available.
            lat = packet.get("latitude", None)
            lon = packet.get("longitude", None)
            alt = packet.get("altitude", None)

            # If we have latitude and longitude, return them.
            if lat is not None and lon is not None:
                return lat, lon, alt
        except Exception as e:
            # Print error and continue
            print("Error parsing APRS data:", e)
            continue


def init_bno085():
    """
    Initialize the BNO085 sensor over I2C using the Adafruit CircuitPython BNO08X library.

    Returns
    -------
    sensor : adafruit_bno08x.BNO08X
        The initialized BNO085 sensor object.
    """
    # Create I2C bus.
    i2c = busio.I2C(board.SCL, board.SDA)

    # Initialize the BNO085 sensor.
    sensor = adafruit_bno08x.BNO08X(i2c)

    # (Optional) Configure sensor settings here if desired.
    return sensor


def read_bno085(sensor):
    """
    Read sensor fusion data from the BNO085.

    The sensor typically provides several types of data. This function retrieves:
      - quaternion: sensor orientation as (w, x, y, z)
      - acceleration: 3-axis acceleration (m/s²)
      - gyro: angular velocity (rad/s)
      - magnetometer: magnetic field vector (microtesla, µT)
      - euler: Euler angles if available (roll, pitch, yaw) in radians

    Returns
    -------
    data : dict
        Dictionary containing sensor measurements.
        For example:
            data = {
                "quaternion": (w, x, y, z),
                "acceleration": (ax, ay, az),
                "gyro": (gx, gy, gz),
                "magnetometer": (mx, my, mz),
                "euler": (roll, pitch, yaw)
            }
    """
    # Read the quaternion, which is returned as (w, x, y, z)
    quat = sensor.quaternion
    # Read acceleration in m/s²
    accel = sensor.acceleration
    # Read gyro data in rad/s
    gyro = sensor.gyro
    # Read magnetometer data in µT
    mag = sensor.magnetic
    # Optionally read Euler angles (roll, pitch, yaw) in radians
    euler = sensor.euler if hasattr(sensor, 'euler') else None

    data = {
        "quaternion": quat,
        "acceleration": accel,
        "gyro": gyro,
        "magnetometer": mag,
        "euler": euler
    }
    return data



# ---------- Replace wrldmagm with ppigrf ----------
def wrldmagm(lat, lon, alt, date):
    """
    Calculate the world magnetic field at a location using ppigrf.

    Inputs:
      lat  - latitude in degrees north.
      lon  - longitude in degrees east.
      alt  - altitude in kilometers above sea level.
      date - a datetime object.

    Uses ppigrf.igrf(lon, lat, alt, date) which returns (Be, Bn, Bu) in the ENU frame.

    Returns:
      decl    - magnetic declination in degrees.
      incl    - magnetic inclination in degrees.
      F_total - total magnetic field strength (nT).
      H_val   - horizontal field magnitude (nT).
      X, Y, Z - magnetic field components in the ENU frame (nT),
                where X = Be (east), Y = Bn (north), Z = Bu (up).
    """
    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt, date)
    F_total = np.sqrt(Be ** 2 + Bn ** 2 + Bu ** 2)
    H_val = np.sqrt(Be ** 2 + Bn ** 2)
    decl = np.degrees(np.arctan2(Be, Bn))
    incl = np.degrees(np.arctan2(Bu, H_val))
    return decl, incl, F_total, H_val, Be, Bn, Bu


# ---------- Helper functions ----------

def skew(v):
    """Return the 3x3 skew-symmetric matrix for a 3-element vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def L_matrix(q, omega):
    """
    Compute the 4x3 left multiplication matrix for quaternion derivative
    for quaternions in vector-then-scalar form.

    Let q = [v, w] where v is 3x1 and w is scalar.
    Then, dq/dt = 0.5 * L(q) * omega, where:
       L(q) = [ w*I_3 + skew(v) ]
              [ -v^T         ]
    Returns a 4x3 matrix.
    """
    v = q[:3]
    w = q[3]
    return np.vstack((w * np.eye(3) + skew(v), -v.reshape(1, 3)))


def quat_deriv(q, omega):
    """
    Compute quaternion derivative dq/dt = 0.5*q⊗[omega,0] for quaternions
    in vector-then-scalar form (q = [q1, q2, q3, q0]).

    Returns a 4-element vector.
    """
    return 0.5 * (L_matrix(q, omega) @ omega)


def computeA_quat(x, I, u):
    """
    Compute the Jacobian of the state derivative with respect to the quaternion part.
    x: state vector [q; omega] (7,).
    Returns a 7x4 matrix.
    If epsilon is provided, use finite differences; otherwise, return the analytic result.
    For quaternions in vector-then-scalar form, the analytic derivative for the quaternion dynamics is:
         d/dq [0.5*L(q)*omega] = [0.5*(dL/dq)*omega],
    where L(q) = [ w*I + skew(v); -v^T ] with q = [v, w].
    A short derivation gives:
         A_q = [ 0.5*skew(omega),  0.5*omega.reshape(3,1) ]
         [ -0.5*omega^T,       0 ]
    and A_omega (the derivative of omega dynamics w.r.t. q) is zero.
    """
    q = x[:4]
    omega = x[4:7]

    # For quaternion dynamics:
    A_q_top = 0.5 * skew(omega)  # 3x3 block
    A_q_bottom = -0.5 * omega.reshape(1, 3)  # 1x3 block
    # Derivative with respect to scalar part (w):
    A_q_top_w = 0.5 * omega.reshape(3, 1)  # 3x1 block
    A_q_bottom_w = np.array([[0]])  # 1x1
    # Assemble the 4x4 analytic Jacobian for dq/dt with respect to q:
    # Structure: columns correspond to perturbations in [v; w]
    A_q = np.hstack((np.vstack((A_q_top, A_q_bottom)), np.vstack((A_q_top_w, A_q_bottom_w))))
    # Angular velocity dynamics do not depend on q:
    A_omega = np.zeros((3, 4))
    A = np.vstack((A_q, A_omega))

    return A

I     = np.eye(3)
I_inv = np.linalg.inv(I)

def computeA_full(x, I, u):
    q     = x[:4]
    omega = x[4:]
    # 1) top-left 4×4 block (∂(dq/dt)/∂q):
    A_q = computeA_quat(x, I, u)[:4,:4]   # if you already have it
    # 2) top-right 4×3 block (∂(dq/dt)/∂ω):
    L = L_matrix(q, omega)               # 4×3
    A_qomega = 0.5 * L
    # 3) bottom-left 3×4 block is zero:
    zeros_3_4 = np.zeros((3,4))
    # 4) bottom-right 3×3 block:
    Iw = I @ omega
    A_omegaomega = I_inv @ (-skew(Iw) - skew(omega) @ I)
    # assemble the full 7×7
    top    = np.hstack((A_q,      A_qomega))
    bottom = np.hstack((zeros_3_4, A_omegaomega))
    return np.vstack((top, bottom))


def compute_external_torque(v_body, wind_body, alt, phase, params, r_balloon, r_payload):
    """
    Compute external torque on gondola from aerodynamic and buoyant forces.

    Arguments
    ---------
    v_body : (3,) array
        Body-frame velocity [vx, vy, vz] (m/s).
    wind_body : (3,) array
        Body-frame wind [Vwx, Vwy, Vwz] (m/s).
    alt : float
        Altitude (m).
    phase : int
        0=ascent, 1=drogue, 2=main.
    params : dict
        Contains keys:
          'rho_air', 'g',
          'Cd_balloon','Cd_payload','A_payload',
          'A_drogue','Cd_drogue',
          'A_parachute','Cd_parachute',
          'mass_total','mass_descent','burstAltitude','mainChuteAltitude'
    r_balloon, r_payload : (3,) arrays
        Lever arms from COM to balloon‐drag and payload‐drag centers.

    Returns
    -------
    tau_ext : (3,) array
        External torque [τx, τy, τz] in body frame (N·m).
    """
    # 1) Determine buoyancy & drag coefficients per phase
    rho = params['rho_air']
    g = params['g']
    if phase == 0: # change to be based on altitude or define phase
        # ascent
        # volume from ideal gas: Vb = (m g R T)/(M_He P) — assume known in params or precomputed
        Vb = (self.params["mg"] * self.params["Ru"] * T) / (self.params["M_He"] * P)
        r = (3 * Vb / (4 * np.pi))(1 / 3)
        Ab = np.pi * r2
        F_buoy = np.array([0, 0, rho * Vb * g])
        CdA_balloon = params['Cd_balloon'] * params['A_balloon']
        CdA_payload = params['Cd_payload'] * params['A_payload']
        CdA = CdA_balloon + CdA_payload
        m = params['mass_total']
    elif phase == 1:
        # drogue
        F_buoy = np.zeros(3)
        CdA = params['Cd_drogue'] * params['A_drogue']
        m = params['mass_descent']
    else:
        # main
        F_buoy = np.zeros(3)
        CdA = params['Cd_parachute'] * params['A_parachute']
        m = params['mass_descent']

    # 2) Relative wind and drag force
    v_rel = v_body - wind_body
    speed = np.linalg.norm(v_rel)
    if speed > 1e-6:
        F_drag = -0.5 * rho * CdA * speed * v_rel
    else:
        F_drag = np.zeros(3)

    # 3) Weight
    F_weight = np.array([0, 0, -m * g])

    # 4) Torques from balloon‐drag and payload‐drag
    #    buoyancy & weight assumed through COM => zero torque
    tau_b = np.cross(r_balloon, F_drag * (CdA_balloon / CdA))
    tau_p = np.cross(r_payload, F_drag * (CdA_payload / CdA))
    return tau_b + tau_p


def state_derivative(x, I, u, F=None, env_state=None, params=None, r_b=None, r_p=None):
    """
    Compute the derivative of the state x = [q; omega], and optionally of the
    state-transition matrix F via dF/dt = A(x) * F.

    Parameters
    ----------
    x : array_like, shape (7,)
        The current state: quaternion (4,) + angular velocity (3,) in vector-then-scalar form.
    I : array_like, shape (3,3)
        The inertia matrix.
    u : array_like, shape (3,)
        The applied torque.
    F : array_like, shape (7,7), optional
        The current state-transition matrix.  If provided, this function will also
        compute dF/dt = A(x) @ F and return it.

    Returns
    -------
    dx : ndarray, shape (7,)
        The state derivative [dqdt; domegadt].
    dF : ndarray, shape (7,7), optional
        The derivative of the state-transition matrix, only if F was passed in.
    """
    # split state
    q = x[:4]
    omega = x[4:7]

    if env_state and params and (r_b is not None) and (r_p is not None):
        tau_ext = compute_external_torque(
            v_body=env_state.get('v_body'),
            wind_body=env_state.get('wind_body'),
            alt=env_state.get('alt'),
            phase=env_state.get('phase'),
            params=params,
            r_balloon=r_b,
            r_payload=r_p
        )
    else:
        tau_ext = np.zeros(3)

    # total torque = control + external
    tau_total = u + tau_ext

    # quaternion derivative
    dqdt = quat_deriv(q, omega)   # 0.5 * L(q) @ omega

    # angular velocity derivative
    domegadt = np.linalg.solve(I, tau_total - np.cross(omega, I @ omega))

    dx = np.concatenate((dqdt, domegadt))

    if F is not None:
        A_full = computeA_full(x, I, u)  # now 7×7
        dF = A_full @ F

        return dx, dF

    return dx



def propagate_state_RK4(x, dt, I, u):
    """
    Propagate state x = [q; omega] one time step dt using 4th-order Runge-Kutta.
    Normalizes the quaternion (first 4 elements) after each step.
    """
    k1 = state_derivative(x, I, u)
    x_temp = x + 0.5 * dt * k1
    x_temp[:4] /= np.linalg.norm(x_temp[:4])
    k2 = state_derivative(x_temp, I, u)
    x_temp = x + 0.5 * dt * k2
    x_temp[:4] /= np.linalg.norm(x_temp[:4])
    k3 = state_derivative(x_temp, I, u)
    x_temp = x + dt * k3
    x_temp[:4] /= np.linalg.norm(x_temp[:4])
    k4 = state_derivative(x_temp, I, u)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[:4] /= np.linalg.norm(x_next[:4])
    return x_next



def jacobian_m_body(q, m_ref):
    """
    Compute the analytic Jacobian (3x4) of the rotated magnetic field in the body frame
    with respect to the quaternion components.
    Here q is in vector-then-scalar form: [q1, q2, q3, q0].
    The rotation from ENU to body is obtained from the transpose of the standard rotation matrix.
    For q = [x, y, z, w] and m_ref = [m_x, m_y, m_z], the rotated magnetic field is:
         m_body = R_enu_to_body(q) * m_ref.
    The resulting Jacobian J (3x4) has entries:
         J(i,j) = ∂(m_body)_i/∂q_j.
    The derived expressions (with scalar last) are:

    For f1:
      ∂f1/∂x = 2*x*m_x - 2*z*m_y + 2*y*m_z
      ∂f1/∂y = 2*y*m_x + 2*z*m_y + 2*x*m_z
      ∂f1/∂z = -2*z*m_x + 2*y*m_y - 2*x*m_z
      ∂f1/∂w = -2*y*m_x - 2*x*m_y + 2*w*m_z   (Note: expressions need to be derived carefully.)

    For clarity, we provide a derived form below.
    """
    # Let q = [x, y, z, w]
    x, y, z, w = q
    m_x, m_y, m_z = m_ref
    # Derived Jacobian entries (these expressions follow from the standard formulas):
    # Note: One standard rotation matrix for scalar-last quaternion is:
    # R = [[1-2*(y**2+z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
    #      [2*(x*y + z*w),       1-2*(x**2+z**2),   2*(y*z - x*w)],
    #      [2*(x*z - y*w),       2*(y*z + x*w),     1-2*(x**2+y**2)]]
    # Then m_body = R.T * m_ref.
    # We compute the partial derivatives of m_body with respect to q.
    # For brevity, we provide the following analytic expressions:

    # Partial derivatives of R.T * m_ref yields the 3x4 matrix J:
    J = np.zeros((3, 4))
    # Using symbolic derivation one obtains:
    # J[:,0] (derivative with respect to x):
    J[
        0, 0] = 4 * x * m_x + 2 * y * m_y + 2 * z * m_z - 2 * w * m_y + 2 * w * m_z  # placeholder expression (adjust as needed)
    # For brevity, and because full symbolic derivation is lengthy,
    # we assume the following structure derived similarly as in MATLAB code but adjusted for scalar-last:
    # (Below we provide a version analogous to the earlier analytic Jacobian but with indices shifted)
    J[0, 0] = 2 * x * m_x + 2 * w * m_y - 2 * z * m_z
    J[0, 1] = 2 * y * m_x + 2 * z * m_y + 2 * w * m_z
    J[0, 2] = -2 * z * m_x + 2 * y * m_y - 2 * x * m_z
    J[0, 3] = -2 * w * m_x + 2 * x * m_y + 2 * y * m_z

    J[1, 0] = -2 * w * m_x + 2 * x * m_y + 2 * y * m_z
    J[1, 1] = 2 * z * m_x - 2 * y * m_y + 2 * x * m_z
    J[1, 2] = 2 * y * m_x + 2 * z * m_y + 2 * w * m_z
    J[1, 3] = -2 * x * m_x - 2 * w * m_y + 2 * z * m_z

    J[2, 0] = 2 * z * m_x - 2 * y * m_y + 2 * x * m_z
    J[2, 1] = 2 * w * m_x - 2 * x * m_y - 2 * y * m_z
    J[2, 2] = 2 * x * m_x + 2 * w * m_y - 2 * z * m_z
    J[2, 3] = 2 * y * m_x + 2 * z * m_y + 2 * w * m_z
    return J


def quat2rotmat(q):
    """
    Convert a quaternion q (4-element, with scalar last: [q1, q2, q3, q0]) to a rotation matrix.
    Returns (R, R_E2B) where R is the rotation matrix from body frame to Earth frame,
    and R_E2B is its transpose (for transforming vectors from Earth to body frame).
    """
    x, y, z, w = q  # scalar last
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return R, R.T


def quat2eul(q):
    """
    Convert a quaternion (4-element, scalar last: [q1, q2, q3, q0]) to Euler angles (yaw, pitch, roll) in radians.
    Using the ZYX convention.
    """
    x, y, z, w = q
    # yaw (psi)
    yaw = np.arctan2(2 * (x * y + w * z), w * w + x * x - y * y - z * z)
    # pitch (theta)
    sinp = -2 * (x * z - w * y)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    # roll (phi)
    roll = np.arctan2(2 * (y * z + w * x), w * w - x * x - y * y + z * z)
    return np.array([yaw, pitch, roll])


def euler_to_quat(yaw, pitch, roll):
    """
    Convert Euler angles (yaw, pitch, roll) in radians to a quaternion.

    The Euler angles are assumed to be given in the ZYX (yaw-pitch-roll) order,
    and the returned quaternion is in vector-then-scalar format:

          q = [q1, q2, q3, q0]

    where q0 is the scalar part.

    Parameters
    ----------
    yaw : float
        Rotation around the Z-axis (in radians).
    pitch : float
        Rotation around the Y-axis (in radians).
    roll : float
        Rotation around the X-axis (in radians).

    Returns
    -------
    np.ndarray
        A 4-element numpy array representing the quaternion [q1, q2, q3, q0].
    """
    # Compute half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Quaternion computation for ZYX rotation, scalar last:
    # q = [qx, qy, qz, qw]
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    # Return quaternion in vector-then-scalar order
    return np.array([qx, qy, qz, qw])


def enu2body(q, m_ref):
    """
    Transforms a vector from the local ENU frame to the body frame using a quaternion.

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion in vector-then-scalar format: [q1, q2, q3, q0],
        where q1, q2, q3 are the vector part and q0 is the scalar part.
    m_ref : array_like, shape (3,)
        The reference vector (e.g. magnetic field) in the ENU frame.

    Returns
    -------
    m_body : ndarray, shape (3,)
        The transformed vector in the body frame, calculated as:
            m_body = R_enu_to_body @ m_ref,
        where R_enu_to_body is the transpose (inverse) of the rotation matrix
        computed from the quaternion.
    """
    # Ensure inputs are numpy arrays.
    q = np.asarray(q).flatten()
    m_ref = np.asarray(m_ref).flatten()

    # Unpack quaternion assuming vector-then-scalar ordering.
    x, y, z, w = q  # x, y, z = vector part; w = scalar part.

    # Compute the rotation matrix from body to ENU using the standard formula:
    R_body_to_enu = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

    # The transformation from ENU to body is given by the transpose (inverse) of R_body_to_enu.
    R_enu_to_body = R_body_to_enu.T

    # Multiply the ENU vector by the transpose to obtain m_body.
    m_body = R_enu_to_body @ m_ref

    return m_body


def propagate_F_rk45(dF, dt, n=7, atol=1e-9, rtol=1e-6):
    """
    Propagate the state-transition matrix F over one step of length dt
    by integrating dF/dt = dF, starting from F(0)=I, using RK45

    Parameters
    ----------
    dF : array_like (n×n) or callable
        - If array_like: treated as a constant n×n matrix so that
              dF/dt = dF (constant).
        - If callable: should have signature dF_current = dF(F_mat, t)
              where F_mat is the current (n×n) matrix and t is time.
    dt : float
        Time step over which to propagate F.
    n : int, optional
        Dimension of F (default 7).
    atol, rtol : float, optional
        Absolute and relative tolerances passed to solve_ivp.

    Returns
    -------
    F_dt : ndarray (n×n)
        The state-transition matrix at time t=dt.
    """
    # initial condition (flattened identity)
    I_flat = np.eye(n).flatten()

    # build the ODE right-hand side
    if callable(dF):
        def rhs(t, F_flat):
            F_mat = F_flat.reshape(n, n)
            dF_mat = dF(F_mat, t)  # user‑supplied function
            return dF_mat.flatten()
    else:
        dF_const = np.asarray(dF)
        if dF_const.shape != (n, n):
            raise ValueError(f"dF must be {(n, n)} or a function, got {dF_const.shape}")

        def rhs(t, F_flat):
            return dF_const.flatten()  # constant derivative

    # integrate from t=0 to t=dt
    sol = solve_ivp(
        rhs,
        t_span=(0.0, dt),
        y0=I_flat,
        method='RK45',
        atol=atol,
        rtol=rtol
    )
    # take the last column of sol.y and reshape
    return sol.y[:, -1].reshape(n, n)


def propagate_state_ivp_all(y0, dt, I, u, atol=1e-6, rtol=1e-3):
    """
    Propagate the combined state+F vector y0 (56 elements) over dt via solve_ivp(RK45).

    y0: array_like, shape (56,)
        First 7 entries are x = [q1,q2,q3,q0, ωx,ωy,ωz],
        next 49 entries are F flattened row-major.

    Returns
    -------
    x_pred   : ndarray (7,)
        Predicted state at t+dt (quaternion renormalized).
    F_pred   : ndarray (7,7)
        Predicted state-transition matrix at t+dt.
    pred_all : ndarray (56,)
        The full flattened [x; F_flat] at t+dt.
    """
    def combined_ode(t, ya):
        # ya: 56-vector
        x_cur = ya[:7]
        F_flat = ya[7:]
        F_cur = F_flat.reshape(7,7)

        # 1) state derivative
        dx = state_derivative(x_cur, I, u)  # returns 7-vector

        # 2) full Jacobian A (7×7)
        A = computeA_full(x_cur, I, u)

        # 3) F derivative, flattened
        dF_flat = (A @ F_cur).reshape(-1)

        return np.concatenate((dx, dF_flat))

    sol = solve_ivp(
        fun=combined_ode,
        t_span=(0.0, dt),
        y0=y0,
        method='RK45',
        atol=atol,
        rtol=rtol
    )

    pred_all = sol.y[:, -1]
    # split back
    x_pred = pred_all[:7]
    F_pred = pred_all[7:].reshape(7,7)
    # renormalize quaternion
    x_pred[:4] /= np.linalg.norm(x_pred[:4])
    return x_pred, F_pred, pred_all # normalize in pred_all


def propagate_state_and_F(y0, dt, I, u, atol=1e-6, rtol=1e-3):
    """
    Propagate both the 7‐state x and the 7×7 transition matrix F over one step dt:
      - uses solve_ivp (RK45) on the 7‐element state
      - uses a single RK4 step on the 7×7 F: dF/dt = A(x) @ F

    Parameters
    ----------
    y0 : array_like, shape (56,)
        Concatenated [x0 (7,), F0_flat (49,)].
    dt : float
        Time step to integrate over.
    I : array_like, shape (3,3)
        Inertia matrix.
    u : array_like, shape (3,)
        Applied torque.
    atol, rtol : float
        Tolerances for state integrator.

    Returns
    -------
    x_pred : ndarray, shape (7,)
        Predicted state at t+dt (quaternion normalized).
    F_pred : ndarray, shape (7,7)
        Predicted transition matrix at t+dt.
    pred_all : ndarray, shape (56,)
        Full concatenated [x_pred, F_pred.flatten()].
    """
    # Split input
    x0 = y0[:7]
    F0_flat = y0[7:]
    F0 = F0_flat.reshape(7, 7)

    # 1) Propagate the 7‐state via RK45
    def ode_state(t, x):
        return state_derivative(x, I, u)

    sol = solve_ivp(
        fun=ode_state,
        t_span=(0.0, dt),
        y0=x0,
        method='RK45',
        atol=atol,
        rtol=rtol
    )
    x_pred = sol.y[:, -1]
    x_pred[:4] /= np.linalg.norm(x_pred[:4])

    # 2) Compute Jacobian at the old state (you may choose x0 or x_pred)
    A = computeA_full(x0, I, u)

    # 3) Propagate F via a single RK4 step
    k1 = A @ F0
    k2 = A @ (F0 + 0.5 * dt * k1)
    k3 = A @ (F0 + 0.5 * dt * k2)
    k4 = A @ (F0 + dt * k3)
    F_pred = F0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # 4) Combine outputs
    pred_all = np.concatenate((x_pred, F_pred.flatten()))
    return x_pred, F_pred, pred_all

# ---------- Main EKF Loop ----------

def main_loop():
    sensor = init_bno085()
    dt = 0.01  # 100 Hz update rate
    # Initialize state vector: [q; omega]
    # q is now [q1, q2, q3, q0] (vector then scalar); start with identity quaternion: [0,0,0,1]
    x_est = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    P = np.eye(7) * 0.1
    Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
    R_meas = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
    I = np.eye(3)  # inertia matrix (placeholder)
    F_init = np.eye(7)

    # Location and date for magnetic field (ppigrf)
    date = datetime(2021, 3, 28)

    while True:

        sensor_data = read_bno085(sensor)
        print("Sensor Data:")
        print("  Quaternion (w, x, y, z):", sensor_data["quaternion"])
        print("  Acceleration (m/s²):", sensor_data["acceleration"])
        print("  Gyro (rad/s):", sensor_data["gyro"])
        print("  Magnetometer (µT):", sensor_data["magnetometer"])
        if sensor_data["euler"] is not None:
            print("  Euler angles (rad):", sensor_data["euler"])
        print("-" * 40)

        lat, lon, alt = read_aprs_gps("/dev/ttyUSB0", 9600)
        w, x, y, z = sensor.quaternion
        x_est = (x, y, z, w)

        # Measurement acquisition
        z_gyro = sensor_data["gyro"]  # 3-element (rad/s)
        z_mag = sensor_data["magnetometer"]  # 3-element magnetometer reading
        # z_accel is not used here.

        # Get magnetic field using ppigrf's igrf:
        decl, incl, F_total, H_val, X, Y, Z = wrldmagm(lat, lon, alt, date)
        m_ref = np.array([X, Y, Z])
        m_ref = m_ref / np.linalg.norm(m_ref)

        # Assemble measurement vector: first 3 are gyro, next 3 are magnetometer.
        z = np.concatenate((z_gyro, z_mag))

        # Extract current state
        q = x_est[:4]
        omega = x_est[4:7]
        # Rotate m_ref (in ENU) to body frame:
        m_body = enu2body(q, m_ref)

        # Applied torque (placeholder)
        u = np.array([0.0, 0.0, 0.0])

        # Prediction step: propagate state using RK4
        dx, dF = state_derivative(x_est, I, u, F=F_init)

        F_mat = propagate_F_rk45(dF, dt, n=7)

        x_est = np.concatenate((x_est, F_mat.flatten()))

        # Suppose you currently have:
        # y0 = np.concatenate((x_est, F_init.flatten()))  # shape (56,)
        x_pred, F_mat, x_est = propagate_state_ivp_all(x_est, dt, I, u)
        # x_pred, F_mat, x_est = propagate_state_and_F(x_est, dt, I, u)

        # Covariance prediction:
        P_pred = F_mat @ P @ F_mat.T + Q

        # Measurement update:
        # Predicted measurement h(x) = [omega_pred; m_body]
        h_gyro = x_pred[4:7]
        h_mag = m_body
        h_x = np.concatenate((h_gyro, h_mag))
        y_res = z - h_x

        # Measurement Jacobian H:
        # For gyro part: derivative w.r.t. state is [zeros(3x4) I_3]
        H_gyro = np.hstack((np.zeros((3, 4)), np.eye(3)))
        # For magnetometer part: compute Jacobian of m_body with respect to q.
        J_m_body = jacobian_m_body(q, m_ref)  # 3x4
        H_mag = np.hstack((J_m_body, np.zeros((3, 3))))
        H_mat = np.vstack((H_gyro, H_mag))

        # Kalman gain:
        S = H_mat @ P_pred @ H_mat.T + R_meas
        K_gain = P_pred @ H_mat.T @ np.linalg.inv(S)

        # Update state estimate:
        x_est = x_pred + K_gain @ y_res
        x_est[:4] /= np.linalg.norm(x_est[:4])
        P = (np.eye(7) - K_gain @ H_mat) @ P_pred

        # Convert quaternion to Euler angles (radians) and print (converted to degrees)
        euler = quat2eul(x_est[:4])
        print("Yaw: {:.2f}°, Pitch: {:.2f}°, Roll: {:.2f}° | ω: [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            np.rad2deg(euler[0]), np.rad2deg(euler[1]), np.rad2deg(euler[2]),
            x_est[4], x_est[5], x_est[6]))

        # ... after you compute x_est and convert to Euler ...
        euler_deg = np.rad2deg(euler)
        send_euler_angles(euler_deg, target_ip="255.255.255.255", target_port=5005)

        q_est = x_est[:4]  # quaternion [qx, qy, qz, qw]
        update_attitude_plot(q_est)


if __name__ == '__main__':
    main_loop()
