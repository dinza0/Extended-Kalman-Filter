# Balloon‑Payload Attitude EKF

Real‑time quaternion EKF with aerodynamic‑torque modelling and live ENU visualisation.

---

## 1 Overview

`Filterv2.py` estimates the attitude (q, ω) of a high‑altitude balloon payload at 100 Hz.
`Filterv2.py` is a test code that operates without hardware and the 
initial states need to be defined. 
`Filterv1.py` is the code to run when hardware and sensors are integrated.

It fuses:

* 3‑axis gyroscope
* 3‑axis magnetometer (IGRF reference)
* Optional aerodynamic torques from buoyancy & drag

The filter runs on a Raspberry Pi 4 (or any desktop) and draws a live 3‑D plot of body axes in the ENU frame.

---

## 2 Folder Structure

```
.
├─ Filterv2.py          ← main EKF + visualiser
├─ balloon_params.json  ← mass, CdA, lever‑arms (edit to your vehicle)
├─ README.md            ← this file
└─ report.md / pdf      ← full technical report
```

---

## 3 Dependencies

```bash
pip install numpy scipy matplotlib ppigrf
#  optional for real hardware:
pip install adafruit-blinka adafruit-circuitpython-bno08x pyserial aprslib
```

* `numpy`, `scipy`   – math & ODE
* `matplotlib`   – 3‑D live plot
* `ppigrf`   – IGRF magnetic model
* `adafruit-bno08x`   – IMU driver (Pi only)
* `pyserial`, `aprslib`   – GPS / APRS tracker (optional)

---

## 4 Quick Start (Desktop Simulation)

1. Clone or copy the repo.
2. Open **`Filterv2.py`**.  At top the BNO085 and GPS blocks are *stubbed* so the code generates a constant 1 rad s⁻¹ yaw spin and synthetic magnetometer field.
3. Run:

   ```bash
   python Filterv2.py
   ```

   You should see terminal output

   ```
   Yaw:  57.3°, Pitch: 0.0°, Roll: 0.0° | ω: [0.00 0.00 1.00] rad/s
   ```

   and a pop‑up 3‑D plot whose red (body‑X) axis slowly yaws about ENU‑Up.

---

## 5 Running on Raspberry Pi with Real Hardware

1. Enable I²C (`sudo raspi-config` → Interfaces).
2. Wire BNO085 to Pi SDA/SCL pins.
3. Install drivers:

   ```bash
   pip install adafruit-blinka adafruit-circuitpython-bno08x
   ```
4. In **`Filterv2.py`** comment‑in `sensor = init_bno085()` and remove the stub lines.
5. Connect GPS/APRS tracker on `/dev/ttyUSB0`; keep `read_aprs_gps` enabled.

---

## 6 Configuration

Edit **`balloon_params`** (or `balloon_params.json`) with your vehicle data:

| Key                                  | Units      | Description                 |
| ------------------------------------ | ---------- | --------------------------- |
| `mass_total`                         | kg         | balloon + payload at launch |
| `mass_descent`                       | kg         | payload + chute after burst |
| `Cd_balloon`, `A_balloon`            | —, m²      | drag coefficient / area     |
| `Cd_payload`, `A_payload`            | —, m²      | payload drag                |
| `A_drogue`, `Cd_drogue`              | m², —      | small chute                 |
| `A_parachute`, `Cd_parachute`        | m², —      | main chute                  |
| `burstAltitude`, `mainChuteAltitude` | m          | phase thresholds            |
| `r_balloon`, `r_payload`             | m 3‑vector | lever arms from COM         |

Add:

```python
from atmosphere import make_Vb_func
balloon_params['Vb_func'] = make_Vb_func(balloon_params)
```

so buoyancy scales with altitude.

---

## 7 Live Visualisation

A Matplotlib 3‑D quiver shows body axes:

* **Red** – Body‑X (East)
* **Green** – Body‑Y (North)
* **Blue** – Body‑Z (Up)

Numeric ticks are hidden for clarity.  Close the window or press `Ctrl‑C` to exit.

---

## 8 Extending the Software

* **Add accelerometer update** – extend measurement vector `z` and Jacobian `H`.
* **Variable dt** – call `propagate_state_and_F()` with runtime‑selected step.
* **Data logging** – insert a CSV writer inside `main_loop()`.
* **Test with Hardware** – the code has not been tested with needed integrated hardware.

---

## 9 Performance Benchmarks

| Propagator                | Pi 4 CPU time / 10 ms step |
| ------------------------- | -------------------------- |
| Full 56‑state `solve_ivp` | 2.10 ms                    |
| Mixed RK45(7)+RK4(49)     | 0.90 ms                    |
| **Full RK4 (56)**         | **0.23 ms**                |

---

## 10 License

MIT License — free to use & modify.  Please cite this repo in derivative works.

---

*Happy flying!* 🚀
