
import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt

g = 9.807
vehicleMass = 262
trackwidth = 1.2
wheelbase = 1.53
b = 0.72675
a = wheelbase - b
h = 0.24
rho_air = 1.2
CdA = 1.1*1.1
CLfA = 0.15*1.1
CLrA = 0.15*1.1
gamma = 50/50
roll_stiffness = 0.53
P_max = 80

ref_load = 1500
pCx1 = 1.532
pDx1 = 2.0217
pDx2 = -1.3356e-12
pEx1 = -0.53967
pKx1 = 31.5328
pKx3 = -0.83511
lambda_mux = 1

pCy1 = 1.5
pDy1 = 2.3298
pDy2 = -0.5
pEy1 = -0.052474
pKy1 = -42.8074
pKy2 = 1.7679
lambda_muy = 1

mu = 1


def mf_fx_fy(kappa, alpha, Fz):
    global ref_load

    error_eps = 1e-6
    # calculate the coefs
    dfz = (Fz - ref_load)/ref_load
    Kx = Fz*pKx1*ca.exp(pKx3*dfz)
    Ex = pEx1
    Dx = (pDx1 + pDx2*dfz)*lambda_mux
    Cx = pCx1
    Bx = Kx/(Cx*Dx*Fz)

    Ky = ref_load*pKy1*ca.sin(2*ca.atan(Fz/(pKy2*ref_load)))
    Ey = pEy1
    Dy = (pDy1 + pDy2*dfz)*lambda_muy
    Cy = pCy1
    By = Ky/(Cy*Dy*Fz)

    # magic formula
    sig_x = kappa/(1 + kappa)
    sig_y = alpha/(1 + kappa)
    sig = ca.sqrt((sig_x**2) + (sig_y**2))

    Fx = Fz*(sig_x/(sig + error_eps))*Dx*ca.sin(Cx * ca.atan(Bx*sig - Ex*(Bx*sig - ca.atan(Bx*sig))))
    Fy = Fz*(sig_y/(sig + error_eps))*Dy*ca.sin(Cy*ca.atan(By*sig - Ey*(By*sig - ca.atan(By*sig))))


    return Fx, Fy

def normal_loads(ax, ay, u):
    FLf = 0.5*CLfA*rho_air*(u**2)
    FLr = 0.5*CLrA*rho_air*(u**2)

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nfl = ca.fmax(Nfl, 10)
    Nfr = ca.fmax(Nfr, 10)
    Nrl = ca.fmax(Nrl, 10)
    Nrr = ca.fmax(Nrr, 10)


    return Nfl, Nfr, Nrl, Nrr

def compute_ax(ay, V, init_vals, maximize):
    global vehicleMass, h, wheelbase, a, b, trackwidth, g, CdA, ClA, rho_air 
    opti = ca.Opti()

    ax = opti.variable()
    delta = opti.variable()
    beta = opti.variable()
    kfl = opti.variable()
    kfr = opti.variable()
    krl = opti.variable()
    krr = opti.variable()

    opti.set_initial(ax, init_vals['ax'])
    opti.set_initial(delta, init_vals['delta'])
    opti.set_initial(beta, init_vals['beta'])
    opti.set_initial(kfl, init_vals['kfl'])
    opti.set_initial(kfr, init_vals['kfr'])
    opti.set_initial(krl, init_vals['krl'])
    opti.set_initial(krr, init_vals['krr'])

    # bounds
    beta_bound = np.pi/4
    delta_bound = np.pi/6
    kappa_bound = 0.3
    ax_bound = 3*g

    opti.subject_to(opti.bounded(-ax_bound, ax, ax_bound))
    opti.subject_to(opti.bounded(-beta_bound, beta, beta_bound))
    opti.subject_to(opti.bounded(-delta_bound, delta, delta_bound))
    opti.subject_to(opti.bounded(-kappa_bound, kfr, kappa_bound))
    opti.subject_to(opti.bounded(-kappa_bound, kfl, kappa_bound))
    opti.subject_to(opti.bounded(-kappa_bound, krl, kappa_bound))
    opti.subject_to(opti.bounded(-kappa_bound, krr, kappa_bound))
    # opti.subject_to(opti.bounded(-kappa_bound, kf, kappa_bound))
    # opti.subject_to(opti.bounded(-kappa_bound, kr, kappa_bound))


    omega = ay/(V*ca.cos(beta))

    u = V*ca.cos(beta)
    v = V*ca.tan(beta)

    lambda_eps = 1e-3
    lambda_fl = delta - ca.atan2((v + omega*a), (u - omega*(trackwidth/2) + lambda_eps))
    lambda_fr = delta - ca.atan2((v + omega*a), (u + omega*(trackwidth/2) + lambda_eps))
    lambda_rl = -ca.atan2((v - omega*b), (u - omega*(trackwidth/2) + lambda_eps))
    lambda_rr = -ca.atan2((v - omega*b), (u + omega*(trackwidth/2) + lambda_eps))

    # normal loads
    Nfl, Nfr, Nrl, Nrr = normal_loads(ax, ay, u)

    fx_fl, fy_fl = mf_fx_fy(kfl, lambda_fl, Nfl)
    fx_fr, fy_fr = mf_fx_fy(kfr, lambda_fr, Nfr)
    fx_rl, fy_rl = mf_fx_fy(krl, lambda_rl, Nrl)
    fx_rr, fy_rr = mf_fx_fy(krr, lambda_rr, Nrr)

    FD = 0.5*CdA*rho_air*(u**2)

    # EOMs
    opti.subject_to(vehicleMass*ax == (fx_fl + fx_fr + fx_rl + fx_rr) - (fy_fl + fy_fr)*delta - FD)
    opti.subject_to(vehicleMass*ay == (fy_fl + fy_fr + fy_rl + fy_rr) + (fx_fl + fx_fr)*delta)

    # power constraint
    # without power constraint = 252 solutions
    # with power constraint = 250 solutions
    opti.subject_to((fx_rl + fx_rr + fx_fl + fx_fr)*u <= P_max*1000)
    
    if maximize:
        opti.minimize(-ax) # maximize acceleration
    else:
        opti.minimize(ax) # braking

    
    opti.solver("ipopt", {
        "ipopt.print_level": 5,
        'expand': True,
        'print_time': False
    }, {
        'max_iter': 2000,
        'tol': 1e-6
    })

    try:
        sol = opti.solve()
        # Check for NaN in solution
        ax_val = sol.value(ax)
        if np.isnan(ax_val):
            return 0, False, None
        sol_vals = {
            'ax': ax_val,
            'delta': sol.value(delta),
            'beta': sol.value(beta),
            'kfl': sol.value(kfl),
            'kfr': sol.value(kfr),
            'krl': sol.value(krl),
            'krr': sol.value(krr)
        }
        return ax_val, True, sol_vals
    except:
        return 0, False, None

V = 15
ay_max = 3*g
N = 150
ay_range = np.linspace(-ay_max, ay_max, N)

ay_values = []
ax_values = []
prev_values = None
for ay in ay_range:
    if prev_values == None:
        init_values = {'ax': 0, 'delta': 0, 'beta': 0, 'kfl': 0, 'kfr': 0, 'krl': 0, 'krr': 0}
    else:
        init_values = prev_values


    ax_val, success, sol_vals = compute_ax(ay, V, init_values, True)


    if success:
        prev_values = sol_vals
        ay_values.append(ay)
        ax_values.append(ax_val)
    else:
        continue

ay_values_brake = []
ax_values_brake = []
for ay in reversed(ay_range):
    if prev_values == None:
        init_values = {'ax': -1.0, 'delta': 0, 'beta': 0, 'kfl': -0.1, 'kfr': -0.1, 'krl': -0.1, 'krr': -0.1}
    else:
        init_values = prev_values


    ax_val, success, sol_vals = compute_ax(ay, V, init_values, False)


    if success:
        prev_values = sol_vals
        ay_values_brake.append(ay)
        ax_values_brake.append(ax_val)
    else:
        continue

ay_values_combined = ay_values + ay_values_brake
ax_values_combined = ax_values + ax_values_brake
ay_norm = np.array(ay_values_combined)/g
ax_norm = np.array(ax_values_combined)/g

print('Ay values: ', len(ay_values_combined))

fig, ax = plt.subplots(figsize=(16,8))
ax.plot(ay_norm, ax_norm)
ax.set_xlabel("ay/g")
ax.set_ylabel("ax/g")
ax.set_title(f"GG Diagram at V={V} m/s")
fig.show()
plt.show()

