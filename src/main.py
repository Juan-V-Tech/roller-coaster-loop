import numpy as np
from vpython import *
# —───────────────────────────────────────────────────────────────────────────
# Global constants & state
# —───────────────────────────────────────────────────────────────────────────
g = 9.81   # gravitational acceleration (m/s²)
m = 1.0    # mass of cart (kg) for normal-force calculation

ds = 0.01  # fixed arc-length step

# Dynamic parameters (updated in init_sim())
h0, R = 2.5, 1.0
L_ramp = L_buffer = L_loop = L_exit = s_end = 0.0

simulate = False
s = 0.0

# References to VPython objects
graph_energy = None
ke_curve = None
pe_curve = None
tot_curve = None

graph_force = None
n_curve = None
track = None
cart = None

# —───────────────────────────────────────────────────────────────────────────
# Slider display update
def update_displays():
    h_display.text = f" {h_slider.value:.2f}"
    R_display.text = f" {R_slider.value:.2f}"

# —───────────────────────────────────────────────────────────────────────────
# Track geometry helpers
# —───────────────────────────────────────────────────────────────────────────
def r_of_s(s_val):
    # Parameters for the entry curve
    entry_length = L_ramp
    entry_height = h0

    # Center the whole track horizontally at x=0
    total_x = entry_length + 2 * R  # Removed L_exit
    x_offset = -total_x / 2

    if s_val < entry_length:
        # Hermite cubic: y = h0 * (2*(s/L)^3 - 3*(s/L)^2 + 1)
        t = s_val / entry_length
        y = h0 * (2 * t**3 - 3 * t**2 + 1)
        x = s_val
        return vector(x + x_offset, y, 0)
    s2 = s_val - entry_length
    if s2 < L_loop:
        phi = -np.pi/2 + s2/R
        center = vector(entry_length + R + x_offset, R, 0)
        return vector(center.x + R*np.cos(phi), center.y + R*np.sin(phi), 0)
    # After the loop, just stay at the end of the loop (no flat exit)
    end_point = vector(entry_length + 2*R + x_offset, 0, 0)
    return end_point

def theta_of_s(s_val):
    # Slope angle of the track (for normal force)
    entry_length = L_ramp
    if s_val < entry_length:
        # Derivative of Hermite cubic
        t = s_val / entry_length
        dydt = h0 * (6 * t**2 - 6 * t) / entry_length
        dxdt = 1
        return np.arctan2(dydt, dxdt)
    s2 = s_val - entry_length
    if s2 < L_loop:
        phi = -np.pi/2 + s2/R
        return phi + np.pi/2
    return 0.0

def curvature_radius(s_val):
    entry_length = L_ramp
    if s_val < entry_length:
        # Hermite cubic: y = h0 * (2*(s/L)^3 - 3*(s/L)^2 + 1)
        t = s_val / entry_length
        # First derivative y'
        dy_dt = h0 * (6 * t**2 - 6 * t) / entry_length
        dx_dt = 1
        y_prime = dy_dt / dx_dt
        # Second derivative y''
        d2y_dt2 = h0 * (12 * t - 6) / (entry_length ** 2)
        d2x_dt2 = 0
        y_double_prime = d2y_dt2 / (dx_dt ** 2)
        # Curvature k = y'' / (1 + y'^2)^(3/2)
        denom = (1 + y_prime ** 2) ** 1.5
        kappa = y_double_prime / denom if denom != 0 else 0
        if abs(kappa) < 1e-8:
            return 1e8  # Effectively straight
        return 1 / abs(kappa)
    s2 = s_val - entry_length
    if s2 < L_loop:
        return R  # Circular loop
    return 1e8  # After loop, treat as straight (effectively infinite radius)

# —───────────────────────────────────────────────────────────────────────────
# Initialize VPython canvas and UI
# —───────────────────────────────────────────────────────────────────────────
scene = canvas(
    width=500, height=600,
    background=color.white,
    title="Roller-Coaster Loop with Real-Time Graphs"
)
# h₀ slider + display
scene.append_to_caption("Release height h₀ (m): ")
h_slider = slider(min=2, max=9, value=h0, step=0.1, length=200, bind=lambda _: update_displays())
h_display = wtext(text=f" {h0:.2f}")
# R slider + display
scene.append_to_caption("\nLoop radius R (m):   ")
R_slider = slider(min=0.5, max=2.5, value=R, step=0.1, length=200, bind=lambda _: update_displays())
R_display = wtext(text=f" {R:.2f}")
scene.append_to_caption("\n\n")
# —───────────────────────────────────────────────────────────────────────────
# Reset simulation: rebuild track, clear curves, start motion
def init_sim():
    global h0, R, L_ramp, L_buffer, L_loop, L_exit, s_end
    global simulate, s, track, cart
    # read sliders
    h0, R = h_slider.value, R_slider.value
    # recompute lengths
    L_ramp = 0.5 * np.pi * h0
    L_buffer = 1.0 * R
    L_loop = 2.0 * np.pi * R
    L_exit = 5.0
    s_end = L_ramp + L_buffer + L_loop + L_exit
    # hide old objects
    if track: track.visible=False
    if cart: cart.visible=False; cart.clear_trail()
    # clear graphs
    ke_curve.data = []
    pe_curve.data = []
    tot_curve.data = []
    n_curve.data = []
    # draw track behind at z=-0.01
    pts = [r_of_s(si) for si in np.linspace(0, s_end, 500)]
    track = curve(pos=[vector(p.x,p.y,-0.01) for p in pts], radius=0.02, color=color.black)
    # spawn cart at start position
    s = ds
    p = r_of_s(s)
    cart = box(pos=vector(p.x,p.y,0), size=vector(0.1,0.1,0.1), color=color.red,
               make_trail=True, trail_radius=0.015, trail_color=color.red)
    simulate = True

# Run/Reset button
run_btn = button(text="Run / Reset", bind=init_sim)

# —───────────────────────────────────────────────────────────────────────────
# Create graphs aligned to the right of the 3D view
graph_energy = graph(
    title='Energy vs t', xtitle='t (s)', ytitle='Energy (J)',
    width=700, height=250, align='right'
)
ke_curve  = gcurve(graph=graph_energy, color=color.blue, label='KE')
pe_curve  = gcurve(graph=graph_energy, color=color.green, label='PE')
tot_curve = gcurve(graph=graph_energy, color=color.black, label='Total E')

graph_force = graph(
    title='Normal Force vs t', xtitle='t (s)', ytitle='N (N)',
    width=700, height=250, align='right'
)
n_curve = gcurve(graph=graph_force, color=color.red, label='N')

# initial run
init_sim()
update_displays()

# —───────────────────────────────────────────────────────────────────────────
# Main loop: animate and plot in real time
while True:
    rate(60)
    if not simulate:
        continue
    # current speed and height
    y = cart.pos.y
    v = np.sqrt(max(0.0, 2 * g * (h0 - y)))
    # forward tangent direction
    t0 = r_of_s(s)
    t1 = r_of_s(min(s + ds, s_end))
    tangent = norm(t1 - t0)
    # normal force
    theta = theta_of_s(s)
    Rc = curvature_radius(s)
    N = m * (v ** 2 / Rc + g * np.cos(theta))
    # energies
    KE = 0.5 * m * v ** 2
    PE = m * g * y
    # plot graphs
    n_curve.plot(s, N)
    ke_curve.plot(s, KE)
    pe_curve.plot(s, PE)
    tot_curve.plot(s, KE + PE)
    # detach on loop if needed
    if s > L_ramp and N <= 0:
        vel = tangent * v
        while cart.pos.y > 0:
            dt = 0.01
            rate(100)
            vel.y -= g * dt
            cart.pos += vel * dt
        simulate = False
        continue
    # advance along track
    s += ds
    p = r_of_s(s)
    cart.pos = vector(p.x, p.y, 0)
    if s >= s_end:
        simulate = False
        continue