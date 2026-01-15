import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt
import time

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

# constants
m = 262
Iz = 130
wheelbase = 1.53
b = 0.72675
a = wheelbase - b
h = 0.24
rho_air = 1.2
Cd = 1.1
Cs = 0.0
Cl = 0.15
A = 1.1
alpha_max = np.deg2rad(20)
kappa_max = 0.12
P_max = 80 # kW
CLfA = Cl*A
CLrA = Cl*A

vehicleMass = m
trackwidth = 1.2
roll_stiffness = 0.53

V = 10
V_dot = 0
phi = np.deg2rad(0)
k = 0 # m/rad

g = 9.807

"""
Functions for computation
"""
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

def sols_mf_fx_fy(kappa, alpha, Fz):
    global ref_load

    error_eps = 1e-6
    # calculate the coefs
    dfz = (Fz - ref_load)/ref_load
    Kx = Fz*pKx1*np.exp(pKx3*dfz)
    Ex = pEx1
    Dx = (pDx1 + pDx2*dfz)*lambda_mux
    Cx = pCx1
    Bx = Kx/(Cx*Dx*Fz)
    
    Ky = ref_load*pKy1*np.sin(2*np.atan(Fz/(pKy2*ref_load)))
    Ey = pEy1
    Dy = (pDy1 + pDy2*dfz)*lambda_muy
    Cy = pCy1
    By = Ky/(Cy*Dy*Fz)

    # magic formula
    sig_x = kappa/(1 + kappa)
    sig_y = alpha/(1 + kappa)
    sig = np.sqrt((sig_x**2) + (sig_y**2))

    Fx = Fz*(sig_x/(sig + error_eps))*Dx*np.sin(Cx * np.atan(Bx*sig - Ex*(Bx*sig - np.atan(Bx*sig))))
    Fy = Fz*(sig_y/(sig + error_eps))*Dy*np.sin(Cy*np.atan(By*sig - Ey*(By*sig - np.atan(By*sig))))


    return Fx, Fy

def normal_loads(ax, ay, u):
    global CLfA, CLrA, rho_air, vehicleMass, a, b, trackwidth, h, roll_stiffness
    FLf = 0.5*CLfA*rho_air*(u**2)
    FLr = 0.5*CLrA*rho_air*(u**2)

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr        
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nfl = ca.fmax(Nfl, 1e-3)
    Nfr = ca.fmax(Nfr, 1e-3)
    Nrl = ca.fmax(Nrl, 1e-3)
    Nrr = ca.fmax(Nrr, 1e-3)


    return Nfl, Nfr, Nrl, Nrr

def sols_normal_loads(ax, ay, u):
    global CLfA, CLrA, rho_air, vehicleMass, a, b, trackwidth, h, roll_stiffness
    FLf = 0.5*CLfA*rho_air*(u**2)
    FLr = 0.5*CLrA*rho_air*(u**2)

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr        
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nfl = max(Nfl, 1e-3)
    Nfr = max(Nfr, 1e-3)
    Nrl = max(Nrl, 1e-3)
    Nrr = max(Nrr, 1e-3)


    return Nfl, Nfr, Nrl, Nrr

def force_calcs(V, ax, Ay_i, beta_i, kf_i, kr_i, delta):
    global m, a, b, h, g, rho_air, P_max, A, Cd, Cs, Cl
    Ay_i = ca.if_else(ca.fabs(Ay_i) > 0.1, Ay_i, ca.sign(Ay_i)*0.1)
    R = (V**2)/(Ay_i)
    r = V/R

    # normal loads
    Nfl, Nfr, Nrl, Nrr = normal_loads(ax, Ay_i, V)

    # slip angles
    alpha_r = ca.atan2((b/R) - ca.sin(beta_i), ca.cos(beta_i))
    alpha_f = delta - ca.atan2(ca.sin(beta_i) + (a/R), ca.cos(beta_i))

    # lateral and longitudinal tire forces
    fx_fl, fy_fl = mf_fx_fy(kf_i, alpha_f, Nfl)
    fx_fr, fy_fr = mf_fx_fy(kf_i, alpha_f, Nfr)
    fx_rl, fy_rl = mf_fx_fy(kr_i, alpha_r, Nrl)
    fx_rr, fy_rr = mf_fx_fy(kr_i, alpha_r, Nrr)
    
    fx_f = fx_fl + fx_fr
    fy_f = fy_fl + fy_fr

    fx_r = fx_rl + fx_rr
    fy_r = fy_rl + fy_rr

    # we have simplified to the planar case
    # ax_b = (V**2)*ca.sin(beta_i)
    # ay_b = (V**2)*-ca.cos(beta_i)
    # az_b = 0
    ax_b = 0
    ay_b = V*r
    az_b = 0 

    gx_b = 0
    gy_b = 0
    gz_b = g

    # aero forces
    fx_aero = 0.5*rho_air*A*Cd*(V**2)
    fy_aero = 0.5*rho_air*A*Cs*(V**2)
    fz_aero = 0.5*rho_air*A*Cl*(V**2)

    Fx_b = ca.cos(delta)*fx_f - ca.sin(delta)*fy_f + fx_r - fx_aero
    Fy_b = ca.cos(delta)*fy_f + ca.sin(delta)*fx_f + fy_r + fy_aero

    forces = {'R': R, 'fz_fl': Nfl, 'fz_fr': Nfr, 'fz_rl': Nrl, 'fz_rr': Nrr, 'alpha_r': alpha_r, 'alpha_f': alpha_f, 'fx_f': fx_f, 'fy_f': fy_f,
              'fx_r': fx_r, 'fy_r': fy_r, 'ax_b': ax_b, 'ay_b': ay_b, 'az_b': az_b, 'gx_b': gx_b, 'gy_b': gy_b, 'gz_b': gz_b,
              'fx_aero': fx_aero, 'fy_aero': fy_aero, 'fz_aero': fz_aero, 'Fx_b': Fx_b, 'Fy_b': Fy_b}
    return forces

def sols_force_calcs(V, ax, Ay_i, beta_i, kf_i, kr_i, delta):
    global m, a, b, h, g, rho_air, P_max, A, Cd, Cs, Cl
    if abs(Ay_i) > 0.1:
        R = (V**2)/(Ay_i)
    else:
        R = (V**2)/(np.sign(Ay_i)*0.1)
    r = V/R


    # normal loads
    Nfl, Nfr, Nrl, Nrr = sols_normal_loads(ax, Ay_i, V)

    # slip angles
    alpha_r = np.atan2((b/R) - np.sin(beta_i), np.cos(beta_i))
    alpha_f = delta - np.atan2(np.sin(beta_i) + (a/R), np.cos(beta_i))

    # lateral and longitudinal tire forces
    fx_fl, fy_fl = sols_mf_fx_fy(kf_i, alpha_f, Nfl)
    fx_fr, fy_fr = sols_mf_fx_fy(kf_i, alpha_f, Nfr)
    fx_rl, fy_rl = sols_mf_fx_fy(kr_i, alpha_r, Nrl)
    fx_rr, fy_rr = sols_mf_fx_fy(kr_i, alpha_r, Nrr)
    
    fx_f = fx_fl + fx_fr
    fy_f = fy_fl + fy_fr

    fx_r = fx_rl + fx_rr
    fy_r = fy_rl + fy_rr

    # we have simplified to the planar case
    # ax_b = (V**2)*np.sin(beta_i)
    # ay_b = (V**2)*-np.cos(beta_i)
    # az_b = 0

    ax_b = 0
    ay_b = V*r
    az_b = 0

    gx_b = 0
    gy_b = 0
    gz_b = g

    # aero forces
    fx_aero = 0.5*rho_air*A*Cd*(V**2)
    fy_aero = 0.5*rho_air*A*Cs*(V**2)
    fz_aero = 0.5*rho_air*A*Cl*(V**2)

    Fx_b = np.cos(delta)*fx_f - np.sin(delta)*fy_f + fx_r - fx_aero
    Fy_b = np.cos(delta)*fy_f + np.sin(delta)*fx_f + fy_r + fy_aero

    forces = {'R': R, 'fz_fl': Nfl, 'fz_fr': Nfr, 'fz_rl': Nrl, 'fz_rr': Nrr, 'alpha_r': alpha_r, 'alpha_f': alpha_f, 'fx_f': fx_f, 'fy_f': fy_f,
              'fx_r': fx_r, 'fy_r': fy_r, 'ax_b': ax_b, 'ay_b': ay_b, 'az_b': az_b, 'gx_b': gx_b, 'gy_b': gy_b, 'gz_b': gz_b,
              'fx_aero': fx_aero, 'fy_aero': fy_aero, 'fz_aero': fz_aero, 'Fx_b': Fx_b, 'Fy_b': Fy_b}
    return forces

def ocp(delta, V, V_dot, ay_min, ay_max, beta_guess, kf_guess, kr_guess, N=50):
    global m, a, b, h, g, rho_air, P_max, A, Cd, Cs, Cl
    global alpha_max, kappa_max
    
    ax = V_dot
    
    opti = ca.Opti()

    # states
    beta = opti.variable(N)
    kappa_f = opti.variable(N)
    kappa_r = opti.variable(N)

    # controls
    beta_prime = opti.variable(N-1)
    kappa_f_prime = opti.variable(N-1)
    kappa_r_prime = opti.variable(N-1)
    # note: got rid of fz_f and fz_r as controls and instead will compute them...

    # Ay = opti.parameter(N)
    ay_range = np.linspace(ay_min, ay_max, N)

    J = 0
    for i in range(N-1):
        beta_i = beta[i]
        kf_i = kappa_f[i]
        kr_i = kappa_r[i]
        Ay_i = ay_range[i] # TODO: correct?
        Ay_i = ca.if_else(ca.fabs(Ay_i) > 0.1, Ay_i, ca.sign(Ay_i)*0.1)

        # do all of our major calculations
        # inputs: V, ax, Ay_i, beta_i, kf_i, kr_i
        # copy globals
        forces = force_calcs(V, ax, Ay_i, beta_i, kf_i, kr_i, delta)

        # residues
        # Rx_B = V_dot*ca.cos(beta_i) - forces['ax_b'] - (forces['Fx_b']/m)
        # Ry_B = V_dot*ca.sin(beta_i) - forces['ay_b'] - (forces['Fy_b']/m)
        Rx_B = -forces['Fx_b']
        Ry_B = m*Ay_i - forces['Fy_b']

        # add to performance metric J
        J += (Rx_B**2) + (Ry_B**2)

        Mz = a*(forces['fy_f']*ca.cos(delta) + forces['fx_f']*ca.sin(delta)) - b*forces['fy_r']

        # derivatives wrt ay
        d_ay = ay_range[i+1] - ay_range[i]
        if i < (N-2):
            opti.subject_to(beta[i+1] == beta[i] + d_ay*beta_prime[i])
            opti.subject_to(kappa_f[i+1] == kappa_f[i] + d_ay*kappa_f_prime[i])
            opti.subject_to(kappa_r[i+1] == kappa_r[i] + d_ay*kappa_r_prime[i])

        opti.subject_to(opti.bounded(-0.5, beta_i, 0.5))
        opti.subject_to(opti.bounded(-kappa_max, kf_i, 0))
        opti.subject_to(opti.bounded(-kappa_max, kr_i, kappa_max))



        # CONSTRAINTS
        # ignoring vertical force and pitching moment constraints bc we computed directly
        # TODO: yaw moment balance constraint?
        Mz = a*(forces['fy_f']*ca.cos(delta) + forces['fx_f']*ca.sin(delta)) - b*forces['fy_r']
        # opti.subject_to(ca.fabs(Mz) <= 0.01)

        # vehicle power constraint
        opti.subject_to((V*ca.fmax(forces['fx_r'], 0))/(P_max*1000) <= 1)

        # tire and slip angle constraints
        # TODO: do we need to move this outside the for loop?
        # opti.subject_to(opti.bounded(-alpha_max, forces['alpha_r'], alpha_max))
        # opti.subject_to(opti.bounded(-alpha_max, forces['alpha_f'], alpha_max))

        # opti.subject_to(opti.bounded(-kappa_max, kr_i, kappa_max))
        # opti.subject_to(opti.bounded(-kappa_max, kf_i, 0))

    # out of the for loop
    opti.minimize(J)
    
    # set initial guesses
    beta_guess = np.arctan(b/(a+b)*np.tan(delta))
    opti.set_initial(beta, beta_guess)
    opti.set_initial(kappa_f, kf_guess)
    opti.set_initial(kappa_r, kr_guess)

    opti.set_initial(beta_prime, 0.0)
    opti.set_initial(kappa_f_prime, 0.0)
    opti.set_initial(kappa_r_prime, 0.0)

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 2500}
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve()
        beta_sol = sol.value(beta)
        kf_sol = sol.value(kappa_f)
        kr_sol = sol.value(kappa_r)

        mz_vals = []
        residues = []

        # calculating mz and residues and storing them
        for i in range(N):
            beta_i = beta_sol[i]
            kf_i = kf_sol[i]
            kr_i = kr_sol[i]
            Ay_i = ay_range[i]

            forces = sols_force_calcs(V, ax, Ay_i, beta_i, kf_i, kr_i, delta)

            # residues
            # Rx_B = V_dot*np.cos(beta_i) - forces['ax_b'] - (forces['Fx_b']/m)
            # Ry_B = V_dot*np.sin(beta_i) - forces['ay_b'] - (forces['Fy_b']/m)
            Rx_B = -forces['Fx_b']
            Ry_B = m*Ay_i - forces['Fy_b']

            Mz = a*(forces['fy_f']*np.cos(delta) + forces['fx_f']*np.sin(delta)) - b*forces['fy_r']
            mz_vals.append(Mz)

            residue = np.sqrt((Rx_B**2) + (Ry_B**2))
            residues.append(residue)

            if i % 10 == 0 or abs(Ay_i/g) > 1.5:
                print(f"i = {i}, Ay={Ay_i/g:.3f}g, residue={residue:.2f}, mz={Mz:.1f}, beta={np.rad2deg(beta_i):.2f}, kf={kf_i:.4f}, kr={kr_i:.4f}")


        # mz_normalized = np.array(mz_vals)/(m*g*wheelbase)
        mz_normalized = np.array(mz_vals)
        ay_normalized = ay_range/g
        residue_threshold = 50
        residues = np.array(residues)

        return ay_normalized, mz_normalized, beta_sol, residues, kf_sol, kr_sol
            
    except:
        return None, None, None, None, None, None



delta_min = -9
delta_max = 10 # will get us to 9
delta_step = 1
delta_range = np.arange(delta_min, delta_max, delta_step)

V = 15
V_dot = 0
ay_min = -3*g
ay_max = 3*g

results = {}

prev_beta = None
prev_kf = None
prev_kr = None

for delta in delta_range:
    print("delta = ", delta)
    delta_rad = np.deg2rad(delta)
    if prev_beta is not None:
        ay_norm, mz_norm, beta_sol, residues, kf_sol, kr_sol = ocp(delta_rad, V, V_dot, ay_min, ay_max, prev_beta, prev_kf, prev_kr)
    else:
        beta_guess = np.arctan(b/(a+b)*np.tan(delta_rad))
        ay_norm, mz_norm, beta_sol, residues, kf_sol, kr_sol = ocp(delta_rad, V, V_dot, ay_min, ay_max, beta_guess, 0.0, 0.0)


    if ay_norm  is not None:
        ay_norm = np.array(ay_norm)
        mz_norm = np.array(mz_norm)
        beta_sol = np.array(beta_sol)
        residues = np.array(residues)

        results[delta] = {
                'ay': ay_norm,
                'mz': mz_norm,
                'beta': beta_sol,
                'residues': residues
        
        }
        prev_beta = beta_sol
        prev_kf = kf_sol
        prev_kr = kr_sol
    else:
        print("{} failed".format(delta))

   
    # if ay_norm is not None:
    #     threshold = 0.1
    #     valid = residues < threshold
    #     print("VALID: ", valid)

    #     ay_norm = np.array(ay_norm)
    #     mz_norm = np.array(mz_norm)
    #     beta_sol = np.array(beta_sol)
    #     residues = np.array(residues)

    #     print('MZ NORM: ', mz_norm)
    #     
    #     try:
    #         results[delta] = {
    #                 'ay': ay_norm[valid],
    #                 'mz': mz_norm[valid],
    #                 'beta': beta_sol[valid],
    #                 'residues': residues[valid]
    #         }
    #         print("{} DONE".format(delta))
    #     except:
    #         print("{} has no residues under the threshold".format(delta))
    # else:
    #     print("{} failed".format(delta))


# only keep points if the solver did not fail (high residues = weird result = failure)
# filtered_results = {k: v for k, v in results.items() if np.all(v['residues'] < 30)}

# plot
fig, ax = plt.subplots(figsize=(14,6))

for delta, data in results.items():
    delta_deg = np.round(np.rad2deg(delta), decimals=2)
    residues = data['residues']
    mask = residues <= 30
    if len(data['ay']) > 0:
        color = None
        if delta_deg > 0:
            color = 'red'
        elif delta_deg == 0:
            color = 'black'
        else:
            color = 'blue'

        ax.plot(data['ay'][mask], data['mz'][mask], '-', color=color, linewidth=1.5)

ax.set_xlabel("Normalized lateral accel (g)", fontsize=12)
ax.set_ylabel("Yaw moment", fontsize=12)
ax.set_title(f"Yaw Moment Diagram for V={V} m/s")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
