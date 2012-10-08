#!/usr/bin/env python
import numpy as np
from sys import stdout
import os

class MoI:
    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    Set-up:


              SALINE -> sigma_3 = [sigma_3x, sigma_3y, sigma_3z]

    <----------------------------------------------------> x = + a
    
              TISSUE -> sigma_2 = [sigma_2x, sigma_2y, sigma_2z]


                   o -> charge_pos = [x,y,z]


    <-----------*----------------------------------------> x = -a               
                 \-> elec_pos = [x,y,z] 

                 ELECTRODE -> sigma_3 = [sigma_3x, sigma_3y, sigma_3z]
        


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
        if len(self.sigma_1) != 3 or len(self.sigma_2) != 3\
           or len(self.sigma_3) != 3:
            raise ValueError("Conductivities sigma_{1,2,3} must be arrays of length 3")
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

    def in_domain(self, elec_pos, charge_pos):
        """ Checks if elec_pos and charge_pos is within valid area.
        Otherwise raise exception."""
        if not (-self.a <= elec_pos[0] <= self.a):
            print "Electrode position: ", elec_pos
            raise RuntimeError("Electrode not within valid range")
        if not (-self.a <= charge_pos[0] <= self.a):
            print "Charge position: ", charge_pos
            raise RuntimeError("Charge not within valid range")
        dist = np.sqrt( np.sum( (np.array(charge_pos) - np.array(elec_pos))**2 ))
        if dist < 1e-6:
            print "Charge position: ", charge_pos, "Electrode position: ", elec_pos
            raise RuntimeError("Charge and electrode at same position!")

    def anisotropic_moi(self, charge_pos, elec_pos, imem=1):
        """ This function calculates the potential at the position elec_pos = [x,y,z]
        set up by the charge at position charge_pos = [x,y,z]. To get get the potential
        from multiple charges, the contributions must be summed up.
        
        """
        self.in_domain(elec_pos, charge_pos) # Check if valid positions
        factor_lower = 1
        factor_upper = 1
        delta = 1
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        n = 1
        phi = self._R(self.sigma_2, x - x0, y - y0, z - z0)

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
            phi += delta
            n += 1
        phi *= imem/(4*np.pi)
        return phi

    def potential_at_elec(self, charge_pos, elec_pos, r, imem=1):
        """ Calculate the potential at electrode 'elec_index' """
        elec_pos_1 = [elec_pos[0], elec_pos[1] + r, elec_pos[2]]
        elec_pos_2 = [elec_pos[0], elec_pos[1] - r, elec_pos[2]]
        elec_pos_3 = [elec_pos[0], elec_pos[1], elec_pos[2] + r]
        elec_pos_4 = [elec_pos[0], elec_pos[1], elec_pos[2] - r]
        phi_center = self.anisotropic_moi(charge_pos, elec_pos, imem)    
        phi_1 = self.anisotropic_moi(charge_pos, elec_pos_1, imem)    
        phi_2 = self.anisotropic_moi(charge_pos, elec_pos_2, imem)
        phi_3 = self.anisotropic_moi(charge_pos, elec_pos_3, imem)
        phi_4 = self.anisotropic_moi(charge_pos, elec_pos_4, imem)
        return (phi_center + phi_1 + phi_2 + phi_3 + phi_4)/5

    def make_mapping(self, neur_dict, ext_sim_dict):
        """ Make a mapping given two arrays of electrode positions"""
        print '\033[1;35mMaking mapping for %s...\033[1;m' %neur_dict["name"]
        neur_input = os.path.join(ext_sim_dict['neural_input'],\
                                         neur_dict['name'], 'coor.npy')
        comp_coors = np.load(neur_input)
        n_compartments = len(comp_coors[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        mapping = np.zeros((n_elecs,n_compartments))
        steps = ext_sim_dict['moi_steps']
        elec_x = ext_sim_dict['elec_x'] # Scalar
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Array    
        for comp in xrange(n_compartments):
            percentage = (comp+1)*100/n_compartments
            stdout.write("\r%d %% complete" % percentage)
            stdout.flush()
            for elec in xrange(n_elecs):
                elec_pos = [elec_x, elec_y[elec], elec_z[elec]]
                charge_pos = comp_coors[:,comp]
                if ext_sim_dict['include_elec']:
                    mapping[elec, comp] += self.potential_at_elec(\
                        charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                else:
                    mapping[elec, comp] += self.anisotropic_moi(\
                        charge_pos, elec_pos)
        print ''
        np.save(os.path.join(ext_sim_dict['output_folder'], 'mappings', 'map_%s.npy' \
                %(neur_dict['name'])), mapping)
        return mapping

    def find_signal_at_electrodes(self, neur_dict, ext_sim_dict, mapping):
        """ Calculating the potential at the electrodes,
        given the mapping from the make_mapping method."""
        
        print '\033[1;35mFinding signal at electrodes ...\033[1;m'
        neur_input_folder = ext_sim_dict['neural_input'] +\
                            neur_dict['name'] + '/'
        imem =  np.load(neur_input_folder + 'imem.npy')
        ntsteps = len(imem[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        signals = np.zeros((n_elecs, ntsteps))
        n_compartments = len(imem[:,0])
        for elec in xrange(n_elecs):
            for comp in xrange(n_compartments):
                signals[elec,:] += mapping[elec,comp] * imem[comp, :]
            np.save(os.path.join(ext_sim_dict['output_folder'], 'signals', \
                                 'signal_%s.npy' %(neur_dict['name'])), signals)           
        return signals
