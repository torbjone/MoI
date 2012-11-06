#!/usr/bin/env python
import numpy as np
from sys import stdout
import os

class MoI:
    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    Set-up:


              SALINE -> sigma_3 

    <----------------------------------------------------> x = + a
    
              TISSUE -> sigma_2


                   o -> charge_pos = [x,y,z]


    <-----------*----------------------------------------> x = -a               
                 \-> elec_pos = [x,y,z] 

                 ELECTRODE -> sigma_1 
        

    '''
    def __init__(self,
                 set_up_parameters = {
                     'sigma_1': 0.0, # Below electrode
                     'sigma_2': 0.3, # Tissue
                     'sigma_3': 3.0, # Saline
                     'slice_thickness': 0.2,
                     'steps' : 20,
                      },
                 debug = False
                 ):
        self.set_up_parameters = set_up_parameters
        self.sigma_1 = set_up_parameters['sigma_1']
        self.sigma_2 = set_up_parameters['sigma_2']
        self.sigma_3 = set_up_parameters['sigma_3']
        if len(self.sigma_1) == 3 or len(self.sigma_2) == 3\
           or len(self.sigma_3) == 3:
            print "Conductivities sigma_{1,2,3} must be scalars"
            if (sigma_1[0] == sigma_1[1] == sigma_1[2]) and \
               (sigma_2[0] == sigma_2[1] == sigma_2[2]) and \
               (sigma_3[0] == sigma_3[1] == sigma_3[2]):
                self.sigma_1 = self.sigma_1[0]
                self.sigma_2 = self.sigma_2[0]
                self.sigma_3 = self.sigma_3[0]
            else:
                raise ValueError("Can't handle anisotropic yet!")
        self.slice_thickness = set_up_parameters['slice_thickness']
        self.a = self.slice_thickness/2
        self.steps = set_up_parameters['steps']


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
        def _MX_function(self, sigma, p, y_dist, z_dist):
            if sigma[0]*sigma[1]*sigma[2] == 0:
                return 0
            else:
                return sigma[2] * sigma[1] * sigma[2]/\
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
            delta = factor_upper * self._R(self.sigma_2, x_dist_upper, y - y0, z-z0) +\
                    factor_lower * self._R(self.sigma_2, x_dist_lower, y - y0, z-z0)
            #print "\n", elec_pos
            #print "p_upper: ", p_upper 
            #print "p_lower: ",p_lower 
            #print "x_dist_lower: ",x_dist_lower
            #print "x_dist_upper: ",x_dist_upper
            #print "factor_lower: ", factor_lower
            #print "factor_upper: ",factor_upper
            #print "phi",phi
            #print "delta",delta
            phi += delta
            n += 1
        phi *= imem/(4*np.pi)
        return phi


    def isotropic_moi(self, charge_pos, elec_pos, imem=1):
        """ This function calculates the potential at the position elec_pos = [x,y,z]
        set up by the charge at position charge_pos = [x,y,z]. To get get the potential
        from multiple charges, the contributions must be summed up.
        
        """

        def _omega(dx):
            return 1/np.sqrt( (y - y0)**2 + (z - z0)**2 + dx**2) 
        self.in_domain(elec_pos, charge_pos) # Check if valid positions
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        phi = _omega(x - x0)
        n = 0
        W23 = (self.sigma_2 - self.sigma_3)/(self.sigma_2 + self.sigma_3)
        W21 = (self.sigma_2 - self.sigma_1)/(self.sigma_2 + self.sigma_1)
        while n < self.steps:
            if n == 0:
                phi += W23 * _omega(x + x0 - (4*n + 2)*a) +\
                       W21 * _omega(x + x0 + (4*n + 2)*a)
            else:
                phi += (W32*W21)**n *(\
                    W23 * _omega(x + x0 - (4*n + 2)*a) + W21 * _omega(x + x0 + (4*n + 2)*a) +\
                    _omega(x - x0 + 4*n*a) + _omega(x - x0 - 4*n*a) )
            n += 1
        phi *= imem/(4*np.pi*self.sigma_2)
        return phi

    def line_source_moi(self, comp_start, comp_end, comp_length, elec_pos, imem=1):
        """ Calculate the moi line source potential"""
        self.in_domain(elec_pos, comp_start)
        self.in_domain(elec_pos, comp_end)

        x0, y0, z0 = comp_start[:]
        x1, y1, z1 = comp_end[:]
        x, y, z = elec_pos[:]
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        a_z = z - z0
        a_y = y - y0
        def _omega(a_x):
            num = comp_length**2 - a_z * dz - a_y * dy - a_x * dx + \
                  comp_length*np.sqrt((a_x - dx)**2 + (a_y - dy)**2 + (a_z - dz)**2)
            den = - a_z * dz - a_y * dy - a_x * dx + \
                  comp_length*np.sqrt(a_x**2 + a_y**2 + a_z**2)
            return np.log(num/den)
        phi = _omega(-self.a - x0)
        n = 1
        while n < self.steps:
            delta = _omega((4*n-1)*self.a - x0) + _omega(-(4*n+1)*self.a - x0)
            phi += ((self.sigma_2 - self.sigma_3)/(self.sigma_2 + self.sigma_3))**n *delta
            n += 1   
        phi *= 2*imem/(4*np.pi*self.sigma_2 * comp_length)
        return phi

    def point_source_moi_at_elec(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi line source potential"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        def _omega(dx):
            return 1/np.sqrt( (y - y0)**2 + (z - z0)**2 + dx**2) 
        phi = _omega(-self.a - x0)
        n = 1
        while n < self.steps:
            delta = _omega((4*n-1)*self.a - x0) + _omega(-(4*n+1)*self.a - x0)
            phi += ((self.sigma_2 - self.sigma_3)/(self.sigma_2 + self.sigma_3))**n *delta
            n += 1   
        phi *= 2*imem/(4*np.pi*self.sigma_2 * comp_length)
        return phi

    
    def potential_at_elec_line_source(self, comp_start, comp_end, comp_length, elec_pos, r, imem=1):
        """ Calculate the potential at electrode 'elec_index' """
        elec_pos_1 = [elec_pos[0], elec_pos[1] + r, elec_pos[2]]
        elec_pos_2 = [elec_pos[0], elec_pos[1] - r, elec_pos[2]]
        elec_pos_3 = [elec_pos[0], elec_pos[1], elec_pos[2] + r]
        elec_pos_4 = [elec_pos[0], elec_pos[1], elec_pos[2] - r]
        phi_0 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos, imem)    
        phi_1 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_1, imem)    
        phi_2 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_2, imem)
        phi_3 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_3, imem)
        phi_4 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_4, imem)
        return (phi_0 + phi_1 + phi_2 + phi_3 + phi_4)/5

    def potential_at_elec(self, charge_pos, elec_pos, r, imem=1):
        """ Calculate the potential at electrode 'elec_index' """
        elec_pos_1 = [elec_pos[0], elec_pos[1] + r, elec_pos[2]]
        elec_pos_2 = [elec_pos[0], elec_pos[1] - r, elec_pos[2]]
        elec_pos_3 = [elec_pos[0], elec_pos[1], elec_pos[2] + r]
        elec_pos_4 = [elec_pos[0], elec_pos[1], elec_pos[2] - r]
        phi_0 = self.anisotropic_moi(charge_pos, elec_pos, imem)    
        phi_1 = self.anisotropic_moi(charge_pos, elec_pos_1, imem)    
        phi_2 = self.anisotropic_moi(charge_pos, elec_pos_2, imem)
        phi_3 = self.anisotropic_moi(charge_pos, elec_pos_3, imem)
        phi_4 = self.anisotropic_moi(charge_pos, elec_pos_4, imem)
        return (phi_0 + phi_1 + phi_2 + phi_3 + phi_4)/5

    def potential_at_elec_big_average(self, charge_pos, elec_pos, r, imem=1):
        """ Calculate the potential at electrode 'elec_index' with many points"""
        #import pylab as pl
        number_of_points = 0
        splits = 100
        phi = 0
        for n_z in xrange(splits):
            for n_y in xrange(splits):
                e_z = -r + 2*n_z*r/(splits-1)
                e_y = -r + 2*n_y*r/(splits-1)
                if not np.sqrt(e_z**2 + e_y**2) <= r:
                    continue
                number_of_points += 1
                phi += self.anisotropic_moi(charge_pos, [elec_pos[0], e_z, e_y], imem)
        return phi/number_of_points

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
                    if ext_sim_dict['use_line_source']:
                        mapping[elec, comp] += self.potential_at_elec_line_source(\
                            comp_start, comp_end, comp_length, elec_pos, ext_sim_dict['elec_radius'])
                    else:
                        mapping[elec, comp] += self.potential_at_elec(\
                            charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                else:
                    if ext_sim_dict['use_line_source']:
                        mapping[elec, comp] += self.line_source_moi(\
                            comp_start, comp_end, comp_length, elec_pos)
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
        
        print '\033[1;35mFinding signal at electrodes from %s ...\033[1;m' % neur_dict['name']
        neur_input = os.path.join(ext_sim_dict['neural_input'],
                            neur_dict['name'], 'imem.npy')
        imem =  np.load(neur_input)
        ntsteps = len(imem[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        signals = np.zeros((n_elecs, ntsteps))
        n_compartments = len(imem[:,0])
        for elec in xrange(n_elecs):
            for comp in xrange(n_compartments):
                signals[elec,:] += mapping[elec, comp] * imem[comp,:]
        np.save(os.path.join(ext_sim_dict['output_folder'], 'signals', \
                                 'signal_%s.npy' %(neur_dict['name'])), signals)           
        return signals
