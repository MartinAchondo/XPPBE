import numpy as np


def calculate_phiens(N=10):
       #crear dominio circular (cascaron para generalizar)

        rmin = 4.5
        rmax = 10

        N_r = N

        xspace = np.linspace(-rmax, rmax, N_r)
        yspace = np.linspace(-rmax, rmax, N_r)
        zspace = np.linspace(-rmax, rmax, N_r)
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)

        r = np.sqrt(X**2 + Y**2 + Z**2)
        inside1 = r < rmax
        X1 = X[inside1]
        Y1 = Y[inside1]
        Z1 = Z[inside1]
        r = np.sqrt(X1**2 + Y1**2 + Z1**2)
        inside = r > rmin

        X_r = np.vstack([X1[inside].flatten(),Y1[inside].flatten(), Z1[inside].flatten()]).T

        experimental(X_r)


def analytic(r):

    rI = 1
    epsilon_1 = 1
    epsilon_2 = 80
    kappa = 0.125
    q = 1.0

    f_IN = lambda r: (q / (4 * np.pi)) * (1 / (epsilon_1 * r) - 1 / (epsilon_1 * rI) + 1 / (epsilon_2 * (1 + kappa * rI) * rI))
    f_OUT = lambda r: (q / (4 * np.pi)) * (np.exp(-kappa * (r - rI)) / (epsilon_2 * (1 + kappa * rI) * r))

    y = np.piecewise(r, [r <= rI, r > rI], [f_IN, f_OUT])

    return y


def experimental(X):

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12     
    kb = 1.380649e-23              
    Na = 6.02214076e23
    T = 300

    ang_to_m = 1e-10
    to_V = qe/(eps0 * ang_to_m)   
    kT = kb*T
    C = qe/kT                

    R_x = np.linalg.norm(X, axis=1)

    phi = analytic(R_x)

    C_phi = phi * to_V * C

    G2_p = np.sum(np.exp(-C_phi-6*np.log(R_x)))
    G2_m = np.sum(np.exp(C_phi-6*np.log(R_x)))

    phi_ens_pred = -kT/(2*qe) * np.log(G2_p/G2_m) * 1000 

    print(phi_ens_pred)



if __name__=='__main__':
     
    calculate_phiens(100)
