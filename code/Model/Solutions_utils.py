import numpy as np
import tensorflow as tf
from scipy import special as sp


class Solution_utils(): 

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12     
    kb = 1.380649e-23              
    Na = 6.02214076e23
    ang_to_m = 1e-10
    to_V = qe/(eps0 * ang_to_m)  
    cal2j = 4.184

    def phi_known(self,function,field,X,flag,R=None):  
        x, y, z = X[:,0], X[:,1], X[:,2]
        r = np.linalg.norm(X, axis=1)   
           
        if function == 'Harmonic_spheres':
            phi_values = self.Harmonic_spheres(X, flag, R)
        elif function == 'G_Yukawa':
            phi_values = self.G_Yukawa(x,y,z)
        elif function == 'analytic_Born_Ion':
            phi_values = self.analytic_Born_Ion(r)
        
        if field == 'phi':
            return np.array(phi_values)
        elif field == 'react':
            return np.array(phi_values - self.G(x,y,z))


    def G_Yukawa(self,x,y,z):
        sum = 0
        for q_obj in self.q_list:
            qk = q_obj.q
            xk,yk,zk = q_obj.x_q
            r = tf.sqrt((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk*tf.exp(-self.kappa*r)/r
        return (1/(4*self.pi*self.epsilon_2))*sum

    def G(self,x,y,z):
        sum = tf.constant(0, dtype=self.DTYPE)
        for q_obj in self.q_list:
            qk = q_obj.q
            xk,yk,zk = q_obj.x_q
            r = tf.sqrt((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk/r
        return (1/(self.epsilon_1*4*self.pi))*sum
    
    def dG_n(self,x,y,z,n):
        dx = 0
        dy = 0
        dz = 0
        for q_obj in self.q_list:
            qk = q_obj.q
            xk,yk,zk = q_obj.x_q
            r = tf.sqrt((x-xk)**2+(y-yk)**2+(z-zk)**2)
            dg_dr = qk/(r**3) * (-1/(self.epsilon_1*4*self.pi)) * (1/2)
            dx += dg_dr * 2*(x-xk)
            dy += dg_dr * 2*(y-yk)
            dz += dg_dr * 2*(z-zk)
        dg_dn = n[:,0]*dx[:,0] + n[:,1]*dy[:,0] + n[:,2]*dz[:,0]
        return tf.reshape(dg_dn, (-1,1))

    def analytic_Born_Ion(self,r):
        rI = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.q_list[0].q

        f_IN = lambda r: (q/(4*self.pi)) * ( 1/(epsilon_1*r) - 1/(epsilon_1*rI) + 1/(epsilon_2*(1+kappa*rI)*rI) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI)*r))

        y = np.piecewise(r, [r<=rI, r>rI], [f_IN, f_OUT])

        return y

    def analytic_Born_Ion_du(self,r):
        rI = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.q_list[0].q

        f_IN = lambda r: (q/(4*self.pi)) * ( -1/(epsilon_1*r**2) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI))) * (-kappa/r - 1/r**2)

        y = np.piecewise(r, [r<=rI, r>rI], [f_IN, f_OUT])

        return y


    def Harmonic_spheres(self, x, flag, R=None, N=20):

        q = self.qs
        xq = self.x_qs
        E_1 = self.epsilon_1
        E_2 = self.epsilon_2
        kappa = self.kappa
        if R is None:
            R = self.mesh.R_max_dist

        PHI = np.zeros(len(x))

        for K in range(len(x)):
            rho = np.sqrt(np.sum(x[K,:] ** 2))
            zenit = np.arccos(x[K, 2] / rho)
            azim = np.arctan2(x[K, 1], x[K, 0])

            phi = 0.0 + 0.0 * 1j

            for n in range(N):
                for m in range(-n, n + 1):

                    Enm = 0.0
                    for k in range(len(q)):
                        rho_k = np.sqrt(np.sum(xq[k,:] ** 2))
                        zenit_k = np.arccos(xq[k, 2] / rho_k)
                        azim_k = np.arctan2(xq[k, 1], xq[k, 0])

                        Enm += (
                            q[k]
                            * rho_k**n
                            *4*np.pi/(2*n+1)
                            * sp.sph_harm(m, n, -azim_k, zenit_k)
                        )

                    Anm = Enm * (1/(4*np.pi)) * ((2*n+1)) / (np.exp(-kappa*R)* ((E_1-E_2)*n*self.get_K(kappa*R,n)+E_2*(2*n+1)*self.get_K(kappa*R,n+1)))
                    Bnm = 1/(R**(2*n+1))*(np.exp(-kappa*R)*self.get_K(kappa*R,n)*Anm - 1/(4*np.pi*E_1)*Enm)
                    
                    if flag=='molecule':
                        phi += Bnm * rho**n * sp.sph_harm(m, n, azim, zenit)
                    if flag=='solvent':
                        phi += Anm * rho**(-n-1)* np.exp(-kappa*rho) * self.get_K(kappa*rho,n) * sp.sph_harm(m, n, azim, zenit)

            PHI[K] = np.real(phi)
        
        return PHI


    @staticmethod
    def get_K(x, n):
        K = 0.0
        n_fact = sp.factorial(n)
        n_fact2 = sp.factorial(2 * n)
        for s in range(n + 1):
            K += (
                2**s
                * n_fact
                * sp.factorial(2 * n - s)
                / (sp.factorial(s) * n_fact2 * sp.factorial(n - s))
                * x**s
            )
        return K
    
    def solvation_energy_phi_qs(self,phi_q):
        G_solv = 0.5*np.sum(self.qs * phi_q)
        G_solv *= self.to_V*self.qe*self.Na*(10**-3/self.cal2j)   
        return G_solv
