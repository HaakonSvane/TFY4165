interactive_interface = True        # Specific setting for debugging and development of the script. Only used on MacOSX
show_graphs = True                  # Option to turn of the rendering and calculations of the plots
smoothing = 100                     # Smoothing constant for speed vs. quality of the data plots


import numpy as np
import sympy as sp
import matplotlib as mpl
if interactive_interface:
    mpl.use('macosx')
from matplotlib import pyplot as plt
import sys

fontsize = 22                       # Font size of the plots
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,        # Setting other parameters for plotting
             'lines.linewidth': 2, 'lines.markersize': 7,
             'ytick.labelsize': fontsize, 'xtick.labelsize': fontsize,
             'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(newparams)


'''     INFO
Made by:    H. Svane
Date:       10.10.2018


The script uses different defined functions to perform calculations on the data that was acquired
during the experiment on the Einstein-temperature of aluminium. 

'''

# MEASSURED DATA
m_cup =            [3.3735, 3.3737, 3.3739]
m_cup_avg = np.mean(m_cup)

m_wire =            [0.1185, 0.1172, 0.1175]
m_wire_avg = np.mean(m_wire)


m_metal_and_wire =  [7.36175, 7.3676, 7.3573]
m_metal_and_wire_avg = np.mean(m_metal_and_wire)

t_inserted =        262.81
T0 =                [294.95, 296.95, 299.35]
T0_avg = np.mean(T0)
t_end_boil =        325


kopp_r_topp =       0.07/2           #[m]
kopp_r_bunn=        0.039/2          #[m]
kopp_h =            0.08             #[m]



# FETCHING DATA FROM .CSV FILES
lab_data_b = "data_lab_before.csv"
data_b = np.loadtxt(lab_data_b, delimiter=",")
t_b = data_b[:, 0]
m_b = data_b[:, 1]
m_b += (-m_cup_avg)

lab_data_a = "data_lab_after.csv"
data_a = np.loadtxt(lab_data_a, delimiter=",")
t_a = data_a[:, 0]
m_a = data_a[:, 1]

m_a += (-m_metal_and_wire_avg - m_cup_avg)

collected_x = np.append(t_b, t_a)
collected_y = np.append(m_b, m_a)

# LINEAR REGRESSION MODULE
def lin_reg(x, y):
    S_x = np.sum(x)
    S_y = np.sum(y)
    S_xx = np.sum(np.square(x))
    S_yy = np.sum(np.square(y))
    S_xy = np.sum(np.multiply(x,y))
    delta = np.float64(len(x))*S_xx-S_x**2

    b = (S_y*S_xx-S_x*S_xy)/delta
    a = (float(len(x))*S_xy-S_x*S_y)/delta

    S = np.sum(
        np.power(
            np.subtract(
                np.subtract(y, b), a*x),2))
    dY = np.subtract(a*x+b, y)
    if((float(len(x))-2) != 0):
        db = np.sqrt(1/(float(len(x))-2) * S*S_xx/delta)
        da = np.sqrt(float(len(x))/(float(len(x))-2) * S/delta)
    else:
        db = None
        da = None

    return a, b, da, db, dY

lin_before = lin_reg(t_b, m_b)
lin_after = lin_reg(t_a, m_a)
lin_average = lin_reg(np.array([np.min(collected_x), np.max(collected_x)]),np.array([((lin_before[0]+lin_after[0])*np.min(collected_x) + lin_before[1]+lin_after[1])/2, ((lin_before[0]+lin_after[0])*np.max(collected_x) + lin_before[1]+lin_after[1])/2 ]))


# CONSTANTS
L_nitrogen = 2.0e5
molarmasse_Al = 26.9815385
R_joule = 8.3144598
rho_nitrogen = 807


# GIVEN DATA
Tf = 77.0
E_t_beregnet = 301.8001233597952 # 100K iterations of Newtons method yielded this value


# UNCERTAINTIES
delta_T0 = np.std(T0)                # [K]
delta_Tf = 0.5                       # [K]
delta_t_e = 2                        # [s]
delta_mmw = np.std(m_metal_and_wire) # [g]
delta_mw = np.std(m_wire)            # [g]
delta_Mal = 8e-7                     # [g/mol]
delta_ab = lin_before[2]
delta_bb = lin_before[3]
delta_aa = lin_after[2]
delta_ba = lin_after[3]

deltas = {"T_0":delta_T0, "T_f":delta_Tf, "m_w":delta_mw, "m_mw":delta_mmw, "M_al":delta_Mal, "a_b":delta_ab, "b_b":delta_bb, "a_a":delta_aa, "b_a":delta_ba, "t_e":delta_t_e}


# Defining variables/parameters
c_Vm,   R,E_1,theta_E,k_B,T,T_0,T_f,L,m,delta_m,n,r_b,r_t,h,rho_n,a_0,m_mw,m_w,M_al,a_b,b_b,a_a,b_a,t_s,t_e = sp.symbols(
'c_(Vm) R E_1 theta_E k_B T T_0 T_f L m delta_m n r_b r_t h rho_n a_0 m_mw m_w M_al a_b b_b a_a b_a t_s t_e'
)


#        EQUATIONS

# Calculating the amount of mole present in the aluminium block
n = (m_mw - m_w)/M_al

# Used for determining both the Einstein-temperature and the heat from the loss of mass in the boiling nitrogen
f_main = (theta_E * ((1 / (sp.exp(theta_E / T_0) - 1)) - (1 / (sp.exp(theta_E / T_f) - 1))) - L * delta_m / (3 * n * R))

# Formula to determine the cross-sectional area of the nitrogen in the cup given its current mass.
a_0 = (r_t-r_b)/h
f_Area = sp.pi*((3*a_0*m/(sp.pi*rho_n))+r_b**3)**(2/3)

# Formula for calculating the change in total mass when the liquid is boiling
delta_m_total = (a_b * t_s + b_b - (a_a * t_e + b_a))

# Formula for calculating the change in mass from the heat of the environment around the system
delta_m_s = ((a_b + a_a)/2 * t_s + (b_b + b_a)/2) - ((a_b + a_a)/2 * t_e + (b_b + b_a)/2)

# Formula for calculating the heat emitted from the aluminium
Q_al = L*(delta_m_total - delta_m_s)/1000


# Function to calculate the Gauss error with the variables used to calculate theta_e
def deltaF_Theta(formel, interval, name, T_E ):
    deltaF_list = np.zeros(np.shape(interval), dtype = np.float64)
    diff_list = []
    symb = list(formel.free_symbols)
    symb_list = []

    print("Partiellderiverer komponenter for beregning av "+name+" og beregner feil...")

    for i in range(0, len(symb)):
        if str(symb[i]) in deltas:
            diff_list.append(sp.diff(formel, symb[i]))
            symb_list.append(symb[i])

    print(symb_list)

    for w in range(0, len(interval)):

        sys.stdout.write('\r')
        sys.stdout.write("[%-19s] %d%%" % ('=' * int(w/len(interval)*19+1), (1+ 100 * (w)/len(interval))))
        sys.stdout.flush()

        deltaF_list[w] = np.sum([(diff_list[d].subs(
            [(R, R_joule), (T_0, T0_avg), (T_f, Tf), (theta_E, interval[w]), (m_mw, m_metal_and_wire_avg), (m_w, m_wire_avg), (M_al, molarmasse_Al)]) * deltas[str(symb_list[d])]) ** 2 for d in
                                 range(len(symb_list))])
        deltaF_list[w] = np.sqrt(deltaF_list[w])

    print("\n")
    return np.array(deltaF_list, dtype=np.dtype("Float64"))


# Function to calculate the Gauss error with the variables used to calculate Q
def deltaF_Q(formel, interval, name, T_E ):
    deltaF_list = np.zeros(np.shape(interval), dtype = np.float64)
    diff_list = []
    symb = list(formel.free_symbols)
    symb_list = []

    print("Partiellderiverer komponenter for beregning av "+name+" og beregner feil...")

    for i in range(0, len(symb)):
        if str(symb[i]) in deltas:
            diff_list.append(sp.diff(formel, symb[i]))
            symb_list.append(symb[i])

    print(symb_list)

    for w in range(0, len(interval)):

        sys.stdout.write('\r')
        sys.stdout.write("[%-19s] %d%%" % ('=' * int(w/len(interval)*19+1), (1+ 100 * (w)/len(interval))))
        sys.stdout.flush()

        deltaF_list[w] = np.sum([(diff_list[d].subs(
            [(a_b, lin_before[0]), (b_b, lin_before[1]), (a_a, lin_after[0]), (b_a, lin_after[1]), (t_s, t_inserted), (t_e, t_end_boil), (L, L_nitrogen)]) * deltas[str(symb_list[d])]) ** 2 for d in
                                 range(len(symb_list))])
        deltaF_list[w] = np.sqrt(deltaF_list[w])

    print("\n")
    return np.array(deltaF_list, dtype=np.dtype("Float64"))

# Function to plot the graphs of the data taken during the experiment and its average values
def show_plot_graph(x_list, y_list, a_before, b_before, a_after, b_after):
    plt.minorticks_on()
    plt.scatter(x_list[:8], y_list[:8], label="Before boiling", marker="o", s=60, color="sienna")
    plt.scatter(x_list[8:], y_list[8:], label="After boiling", marker="o", s=60, color="teal")
    plt.plot([np.min(x_list), np.max(x_list)], np.multiply([np.min(x_list), np.max(x_list)], a_before) + b_before, linestyle = ":", color = "sienna", linewidth = 3)
    plt.plot([np.min(x_list), np.max(x_list)], np.multiply([np.min(x_list), np.max(x_list)], a_after) + b_after, linestyle = ":", color = "teal", linewidth = 3)
    plt.plot([np.min(x_list), np.max(x_list)], np.multiply([np.min(x_list), np.max(x_list)], (a_before+a_after)/2) + (b_before+b_after)/2, linestyle="-.", color="green", linewidth=3, label="Avarage m(t)")

    plt.axvline(t_inserted, color="darkgray", linestyle="--", linewidth=3)
    plt.text(240, 35, '$t_{s}$', rotation=0, fontsize=24)
    plt.axvline(t_end_boil, color="darkgray", linestyle="--", linewidth=3)
    plt.text(303, 35, '$t_{e}$', rotation=0, fontsize=24)
    plt.legend(loc='upper right', shadow=False)


    plt.xlabel('$t$ [s]')
    plt.ylabel('$m$  [g]')
    plt.show()

# Function to plot and show the graph that was used in determining the total error of the experiment
def show_Q_graph(start_value, end_value, Q, T_E):
    theta_list = []
    x_s2 = np.linspace(start_value,end_value, smoothing)


    f_Q = (sp.solve(f_main, L * delta_m))[0]
    error_f_theta_e = deltaF_Theta(f_Q, x_s2, "theta", T_E)

    f_Q = f_Q.subs([(R, R_joule), (T_0, T0_avg), (T_f, Tf), (m_mw, m_metal_and_wire_avg), (m_w, m_wire_avg), (M_al, molarmasse_Al)])
    for o in np.nditer(x_s2):
        theta_list.append(f_Q.subs([(theta_E, float(o))]))

    error_f_Q = deltaF_Q(Q_al,[start_value,end_value], "Q", Q)

    theta_array = np.array(theta_list, dtype=np.dtype("Float64"))
    e = plt.subplot(1, 1, 1)
    e.minorticks_on()
    plt.xlim(start_value, end_value)
    e.plot(x_s2, theta_list, color="brown", linewidth=3, label="Calculated $\Theta_E$")
    plt.fill_between(x_s2,np.add(theta_array,error_f_theta_e), np.subtract(theta_array,error_f_theta_e), color="blue", alpha=0.5, label="Error of $Q_{Al}(\Theta_E)$ ")
    e.axhline(Q, color="darkgray", linewidth=3, label="Calculated $Q_{Al}$")
    plt.fill_between(x_s2, float(Q)+error_f_Q[0],float(Q)-error_f_Q[0], color="yellow", alpha=0.5, label="Error of $Q_{Al}$")
    plt.axvline(T_E, linewidth= 3, linestyle= "--", color="darkgray")

    # Calculating the total uncertainty of Theta_E
    array = np.subtract(theta_array,error_f_theta_e)
    value = float(Q)+error_f_Q[0]
    idx = (np.abs(array - value)).argmin()
    plt.axvline(x_s2[idx], linewidth=3, linestyle="--", color="darkgray")
    print(T_E-x_s2[idx])

    plt.legend(loc='upper right', shadow=False)


    e.text(289, 1025, '$\Delta \Theta_E$', rotation=0, fontsize=24)


    plt.xlabel('$\Theta_E$ [K]')
    plt.ylabel('$Q$  [J]')

    plt.show()

# Function to plot and show the graph used to determine the area of the liquid as a function of its mass
def show_area_graph(equation, interval):
    A_list = []
    x_s3 = np.linspace(np.max(interval)/1000, np.min(interval)/1000, smoothing)
    f_A = equation.subs([(r_t, kopp_r_topp), (r_b, kopp_r_bunn), (h, kopp_h), (rho_n, rho_nitrogen)])
    for z in np.nditer(x_s3):
        A_list.append(f_A.subs([(m, z)]))


    plt.plot(np.multiply(1000,x_s3), A_list, linewidth = 3, color = "green")
    plt.xlabel("Mass of nitrogen [g]")
    plt.ylabel("Surface area of the liquid level [m$^{2}$]")
    plt.minorticks_on()
    plt.show()

# Function to calculate the errors for the different variables in the equations
def show_relative_error(difflist, deltavariable):
    y_list = []
    for u in range(len(difflist)):
        y_list.append(difflist[u-1].subs([(R, R_joule), (T_0, T0_avg), (T_f, Tf), (theta_E, E_t_beregnet), (m_mw, m_metal_and_wire_avg), (m_w, m_wire_avg), (M_al, molarmasse_Al)]) * deltas[str(deltavariable[u - 1])])

    print(y_list)

Q_al_calc = Q_al.subs([(a_b, lin_before[0]), (b_b, lin_before[1]), (a_a, lin_after[0]), (b_a, lin_after[1]), (t_s, t_inserted), (t_e, t_end_boil), (L, L_nitrogen)])
if show_graphs:
    show_plot_graph(collected_x, collected_y ,lin_before[0],lin_before[1],lin_after[0],lin_after[1])
    show_Q_graph(250,350,Q_al_calc,E_t_beregnet)
    show_area_graph(f_Area, collected_y)



