# Black-Scholes equation:
# Vt = r * V - r * S * Vx - 0.5 * sigma**2 * S**2 * Vxx
#
# Time step (backwards):
# V(t-dt) = V(t)
#          - 0.5 * sigma^2 * S(t)^2 * V_SS(t) * dt
#          - rS(t)V_S(t) * dt
#          + rV(t) * dt

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm # for Black-Scholes analytical formula
import time

T = 1               # time horizon
dt = 1e-04          # initial time step guess
N_T = int(round(T / dt))      # number of time steps
dt = T / N_T                  # adjusted step size
t_values = np.linspace(0, T, N_T+1)

Smax = 35       # maximum stock price to consider
dS = 0.1        # initial spatial step size 
N_S = int(round(Smax / dS))    # number of spatial steps
dS = Smax / N_S    # adjusted spatial step size
S_values = np.linspace(0, Smax, N_S+1)


r = 0.03 # risk-free rate
sigma = 0.2 # volatility
K = 20  # strike price

print("Run with:")
print("T =", T)
print("dt =", dt)
print("N_T =", N_T)

print("Smax =", Smax)
print("dS =", dS)
print("N_S =", N_S)

print("r =", r)
print("sigma =", sigma)
print("K =", K)

# CFL-type condition (Courant–Friedrichs–Lewy) for stability:
# dt < dS^2 / (sigma^2 * Smax^2)
dt_max = (dS**2) / (sigma**2 * Smax**2)
print(f"Maximum stable dt for explicit Euler: {dt_max}\n")

# Grid to build option value V(t, S(t) surface
V = np.zeros((N_S+1, N_T+1))

def payoff_call(S, K):
    return np.maximum(S - K, 0)

# fill boundary at t = T (payoff at expiry time)
V[:, -1] = payoff_call(S_values, K)

# first and second derivative approximations
def first(a, b, c):
    return (c - a) / (2 * dS)   # Three-point centered-difference
    #return (b - a) / dS        # Two-point forward difference

def second(a, b, c):
    return (c - 2 * b + a) / dS**2

# "x = S"
def black_scholes(V, S, Vx, Vxx):
    Vt = r * V - r * S * Vx - 0.5 * sigma**2 * S**2 * Vxx
    return Vt


# Iterate backwards from T = 1 to t = 0
# Note: Vectorized time step ~20 times faster than for-loop
start = time.time()
for t_index in reversed(range(1, N_T+1)):
    Vxx = (V[2:, t_index] - 2*V[1:-1, t_index] + V[0:-2, t_index]) / dS**2
    Vx = (V[2:, t_index] - V[0:-2, t_index]) / (2*dS)
    Vt = r * V[1:-1, t_index] - r * S_values[1:-1] * Vx - 0.5*sigma**2 * S_values[1:-1]**2 * Vxx
    V[1:N_S, t_index-1] = V[1:N_S, t_index] - dt * Vt

    # boundary conditions
    V[0, t_index-1] = 0
    V[-1, t_index-1] = S_values[-1] - K * np.exp(-r * (T - t_values[t_index-1]))

    if (N_T - t_index + 1) % 10000 == 0:
        print("Processed", N_T - t_index + 1, "/", (N_T), "time values")
end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")

# V[index_S, index_t] is price for option at time t, given stock price S
# index_t = t/T * N_T,  index_S = S/Smax * N_S

###############################################
# Analytic values using Black-Scholes formula
###############################################
S_analytic = S_values[1:] # Exclude S = 0 since analytic formula contains log(S0 * 1/K)
t_values_analytic = t_values[:-1] # exclude t = T, since analytic formula contains division by T - t
V_analytic = np.zeros((len(S_analytic), len(t_values_analytic)))

S_grid, t_grid = np.meshgrid(S_analytic, t_values_analytic, indexing="ij")

dplus = (np.log(S_grid / K) + (r + 0.5 * sigma**2) * (T - t_grid)) / (sigma * np.sqrt(T - t_grid))
dminus = dplus - sigma * np.sqrt(T - t_grid)

V_analytic = S_grid * norm.cdf(dplus) - K * np.exp(-r * (T - t_grid)) * norm.cdf(dminus)

# True payoff at t = T known
poff = payoff_call(S_values[1:], K)
V_analytic = np.hstack((V_analytic, np.atleast_2d(poff).T))

# Option price for S = 0 is 0
V_analytic = np.insert(V_analytic, 0, np.zeros(N_T+1), axis=0)
############################################################

errors = V - V_analytic
print("max abs error  = ", np.max(np.abs(errors)))
print("L2 error        = ", np.sqrt(np.mean(errors**2)))
print("mean error      = ", np.mean(errors))

# Plot value of option, V(S, t), for S=Smax
plt.plot(t_values, V[-1, :], color="blue", label="PDE numeric")
plt.plot(t_values, V_analytic[-1, :], color="red", label="Analytic")
plt.xlabel("t")
plt.ylabel("Call price")
plt.title(f"V(S, t), S={Smax}")
plt.legend()
plt.show()

# Plot V(S, t) for S=K (strike)
S_index_K = int(K / Smax * N_S)
plt.plot(t_values, V[S_index_K, :], color="blue", label="PDE numeric")
plt.plot(t_values, V_analytic[S_index_K, :], color="red", label="Analytic")
plt.xlabel("t")
plt.ylabel("Call price")
plt.title(f"V(S, t), S={K}")
plt.legend()
plt.show()

# Plot V(S, t) for t=0
plt.plot(S_values, V[:, 0], color="blue", label="PDE numeric")
plt.plot(S_values, V_analytic[:, 0], color="red", label="Analytic")
plt.xlabel("S")
plt.ylabel("Call price")
plt.title(f"V(S, t), t={0}")
plt.legend()
plt.show()

# Error plot for S=K
plt.plot(t_values, errors[S_index_K, :], color="blue")
plt.xlabel("t")
plt.ylabel("Error")
plt.title(f"Error plot for S={S_values[S_index_K]}")
plt.show()

# Error plot for t=0
plt.plot(S_values, errors[:, 0], color="blue")
plt.xlabel("S")
plt.ylabel("Error")
plt.title(f"Error plot for t=0, K={K}")
plt.show()
