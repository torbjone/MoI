#!/usr/bin/env python
import numpy as np

class MoI:
    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    '''
    def __init__(self,
                 set_up_parameters = {
                     'sigma_1': [0.0, 0.0, 0.0], # Below electrode
                     'sigma_2': [0.3, 0.3, 0.3], # Tissue
                     'sigma_3': [3.0, 3.0, 3.0], # Saline
                     'slice_thickness': 0.2,
                     'steps' : 20,
                      },
                 debug = False
                 ):
        self.set_up_parameters = set_up_parameters
        self.sigma_1 = set_up_parameters['sigma_1']
        self.sigma_2 = set_up_parameters['sigma_2']
        self.sigma_3 = set_up_parameters['sigma_3']
        self.slice_thickness = set_up_parameters['slice_thickness']
        self.a = self.slice_thickness/2
        self.steps = set_up_parameters['steps']

    def _MX_function(self, sigma, p, y_dist, z_dist):
        if sigma[0]*sigma[1]*sigma[2] == 0:
            return 0
        else:
            return sigma[0] * sigma[1] * sigma[2]/\
                   (sigma[1]*sigma[2]*p**2 + \
                    sigma[0]*sigma[2]*y_dist**2 + sigma[0]*sigma[1]*z_dist**2)

    def _W2X(self, sigma_2, sigma_X, p, y_dist, z_dist):
        M2 = self._MX_function(sigma_2, p, y_dist, z_dist)
        MX = self._MX_function(sigma_X, p, y_dist, z_dist)
        W_value =  (M2 - MX)/(M2 + MX)
        return W_value

    def _R(self, sigma, x_dist, y_dist, z_dist):
        return ( sigma[1]*sigma[2] * x_dist**2 +\
                  sigma[0]*sigma[2] * y_dist**2 +\
                  sigma[0]*sigma[1] * z_dist**2)**(-0.5)

    def anisotropic_moi(self, charge_pos, elec_pos, imem = 1):
        """ This function calculates the potential at the position elec_pos = [x,y,z]
        set up by the charge at position charge_pos = [x,y,z]. To get get the potential
        from multiple charges, the contributions must be summed up.
        """
        factor_lower = 1
        factor_upper = 1
        delta = 1
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        n = 1
        phi = self._R(self.sigma_2, x - x0, y - y0, z - z0)
        #print phi
        #while np.abs(delta) >= 1e-3:
        while n < self.steps:
            p_upper = (2*n - 1) * self.a + x0
            p_lower = (2*n - 1) * self.a - x0
            x_dist_lower = (-1)**(n+1)*(-2*n*self.a - x0) - x
            x_dist_upper = (-1)**(n+1)*( 2*n*self.a - x0) - x
            if n%2 == 1: # Odd terms
                factor_lower *= self._W2X(self.sigma_2, self.sigma_1, \
                                     p_lower, y-y0, z-z0)
                factor_upper *= self._W2X(self.sigma_2, self.sigma_3,\
                                     p_upper, y-y0, z-z0)
            else:
                factor_lower *= self._W2X(self.sigma_2, self.sigma_3,\
                                     p_lower, y-y0, z-z0)
                factor_upper *= self._W2X(self.sigma_2, self.sigma_1,\
                                     p_upper, y-y0, z-z0)
            delta = factor_lower * self._R(self.sigma_2, x_dist_lower,\
                                      y - y0, z-z0) \
                   +factor_upper * self._R(self.sigma_2, x_dist_upper, \
                                      y - y0, z-z0)
            #print delta
            phi += delta
            n += 1
        #pl.show()
        phi *= imem/(4*np.pi)
        return phi
