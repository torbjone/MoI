#!/usr/bin/env python
import numpy as np
from sys import stdout
import os
try:
    from ipdb import set_trace
except:
    pass
class MoI:
    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    Set-up:


              SALINE -> sigma_3 

    <----------------------------------------------------> z = + a
    
              TISSUE -> sigma_2


                   o -> charge_pos = [x,y,z]


    <-----------*----------------------------------------> z = -a               
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
        #set_trace()
        self.sigma_1 = set_up_parameters['sigma_1']
        self.sigma_2 = set_up_parameters['sigma_2']
        self.sigma_3 = set_up_parameters['sigma_3']
        try:
            if len(self.sigma_1) == 3 or len(self.sigma_2) == 3\
                   or len(self.sigma_3) == 3:
                print "\nConductivities are tensors!!\n"
            if (self.sigma_1[0] == self.sigma_1[1] == self.sigma_1[2]) and \
               (self.sigma_2[0] == self.sigma_2[1] == self.sigma_2[2]) and \
               (self.sigma_3[0] == self.sigma_3[1] == self.sigma_3[2]):
                print "Isotropic conductivities should be given as scalars!"
                #raise RuntimeError("Isotropic conductivities should be given as scalars!")
            else:
               self.anisotropic = True
        except TypeError:
            self.anisotropic = False
            
        self.slice_thickness = set_up_parameters['slice_thickness']
        self.a = self.slice_thickness/2
        self.steps = set_up_parameters['steps']

    def in_domain(self, elec_pos, charge_pos):
        """ Checks if elec_pos and charge_pos is within valid area.
        Otherwise raise exception."""
        if not np.abs(elec_pos[2] - self.a) <= 1e-14 and\
           not np.abs(elec_pos[2] + self.a) <= 1e-14:
            print "Electrode position: ", elec_pos
            raise RuntimeError("Electrode not within valid range")
        if not (-self.a <= charge_pos[2] <= self.a):
            print "Charge position: ", charge_pos
            raise RuntimeError("Charge not within valid range")
        dist = np.sqrt( np.sum( (np.array(charge_pos) - np.array(elec_pos))**2 ))
        if dist < 1e-6:
            print "Charge position: ", charge_pos, "Electrode position: ", elec_pos
            raise RuntimeError("Charge and electrode at same position!")

    def _anisotropic_moi(self, charge_pos, elec_pos, imem=1):
        """ This function calculates the potential at the position elec_pos = [x,y,z]
        set up by the charge at position charge_pos = [x,y,z]. To get get the potential
        from multiple charges, the contributions must be summed up.
        
        """
        def _MX_function(sigma, x_dist, y_dist, p):
            if sigma[0]*sigma[1]*sigma[2] == 0:
                return 0
            else:
                return sigma[2] * sigma[1] * sigma[2]/\
                       (sigma[1]*sigma[0]*p**2 + \
                        sigma[0]*sigma[2]*y_dist**2 + sigma[2]*sigma[1]*x_dist**2)

        def _W2X(sigma_2, sigma_X, x_dist, y_dist, p):
            M2 = _MX_function(sigma_2, p, y_dist, x_dist)
            MX = _MX_function(sigma_X, p, y_dist, x_dist)
            W_value =  (M2 - MX)/(M2 + MX)
            return W_value

        def _R(sigma, x_dist, y_dist, z_dist):
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
        phi = _R(self.sigma_2, x - x0, y - y0, z - z0)

        while n < self.steps:
            p_upper = (2*n - 1) * self.a + z0
            p_lower = (2*n - 1) * self.a - z0
            z_dist_lower = (-1)**(n+1)*(-2*n*self.a - z0) - z
            z_dist_upper = (-1)**(n+1)*( 2*n*self.a - z0) - z
            if n%2 == 1: # Odd terms
                factor_lower *= _W2X(self.sigma_2, self.sigma_1, \
                                      x-x0, y-y0, p_lower)
                factor_upper *= _W2X(self.sigma_2, self.sigma_3,\
                                     x-x0, y-y0, p_upper)
            else:
                factor_lower *= _W2X(self.sigma_2, self.sigma_3,\
                                     x-x0, y-y0, p_lower)
                factor_upper *= _W2X(self.sigma_2, self.sigma_1,\
                                    x-x0, y-y0, p_upper)
            delta = factor_upper * _R(self.sigma_2, x-x0, y - y0, z_dist_upper) +\
                    factor_lower * _R(self.sigma_2, x-x0, y - y0, z_dist_lower)
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
        def _omega(dz):
            return 1/np.sqrt( (y - y0)**2 + (x - x0)**2 + dz**2) 
        self.in_domain(elec_pos, charge_pos) # Check if valid positions
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        phi = _omega(z - z0)
        n = 0
        W23 = (self.sigma_2 - self.sigma_3)/(self.sigma_2 + self.sigma_3)
        W21 = (self.sigma_2 - self.sigma_1)/(self.sigma_2 + self.sigma_1)
        while n < self.steps:
            if n == 0:
                phi += W23 * _omega(z + z0 - (4*n + 2)*self.a) +\
                       W21 * _omega(z + z0 + (4*n + 2)*self.a)
            else:
                phi += (W23*W21)**n *(\
                    W23 * _omega(z + z0 - (4*n + 2)*self.a) + W21 * _omega(z + z0 + (4*n + 2)*self.a) +\
                    _omega(z - z0 + 4*n*self.a) + _omega(z - z0 - 4*n*self.a) )
            n += 1
        phi *= imem/(4*np.pi*self.sigma_2)
        return phi

    def line_source_moi(self, comp_start, comp_end, comp_length, elec_pos, imem=1):
        """ Calculate the moi line source potential at electrode plane"""
        self.in_domain(elec_pos, comp_start)
        self.in_domain(elec_pos, comp_end)
        x0, y0, z0 = comp_start[:]
        x1, y1, z1 = comp_end[:]
        x, y, z = elec_pos[:]
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        a_x = x - x0
        a_y = y - y0
        def _omega(a_z):
            num = comp_length**2 - a_x * dx - a_y * dy - a_z * dz + \
                  comp_length*np.sqrt((a_z - dz)**2 + (a_y - dy)**2 + (a_x - dx)**2)
            den = - a_x * dx - a_y * dy - a_z * dz + \
                  comp_length*np.sqrt(a_z**2 + a_y**2 + a_x**2)
            return np.log(num/den)
        phi = _omega(-self.a - z0)
        n = 1
        while n < self.steps:
            delta = _omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0)
            phi += ((self.sigma_2 - self.sigma_3)/(self.sigma_2 + self.sigma_3))**n *delta
            n += 1   
        phi *= 2*imem/(4*np.pi*self.sigma_2 * comp_length)
        return phi

    def point_source_moi_at_elec(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi point source potential"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        def _omega(dz):
            return 1/np.sqrt( (y - y0)**2 + (x - x0)**2 + dz**2) 
        phi = _omega(-self.a - z0)
        W = (self.sigma_2 - self.sigma_3)/(self.sigma_2 + self.sigma_3)
        n = 1
        while n < self.steps:
            delta = _omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0)
            phi += W**n * delta
            n += 1   
        phi *= 2*imem/(4*np.pi*self.sigma_2)
        return phi
    
    def ad_hoc_anisotropic(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi point source potential"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        
        W = (np.min(self.sigma_2) - np.min(self.sigma_3))/\
            (np.min(self.sigma_2) + np.min(self.sigma_3))
        
        def _omega(dz):
            return 1/np.sqrt(self.sigma_2[0]*self.sigma_2[2]*(y - y0)**2 + \
                             self.sigma_2[0]*self.sigma_2[1]*dz**2 + \
                             self.sigma_2[1]*self.sigma_2[2]*(x - x0)**2) 
        phi = _omega(-self.a - z0)
        n = 1
        while n < self.steps:
            phi += W**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        phi *= 2*imem/(4*np.pi)
        return phi

    def potential_at_elec_line_source(self, comp_start, comp_end, comp_length, elec_pos, r, imem=1):
        """ Calculate the potential at electrode 'elec_index' """
        elec_pos_1 = [elec_pos[0], elec_pos[1] + r, elec_pos[2]]
        elec_pos_2 = [elec_pos[0], elec_pos[1] - r, elec_pos[2]]
        elec_pos_3 = [elec_pos[0] + r, elec_pos[1], elec_pos[2]]
        elec_pos_4 = [elec_pos[0] - r, elec_pos[1], elec_pos[2]]
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
        elec_pos_3 = [elec_pos[0] + r, elec_pos[1], elec_pos[2]]
        elec_pos_4 = [elec_pos[0] - r, elec_pos[1], elec_pos[2]]
        phi_0 = self.isotropic_moi(charge_pos, elec_pos, imem)    
        phi_1 = self.isotropic_moi(charge_pos, elec_pos_1, imem)    
        phi_2 = self.isotropic_moi(charge_pos, elec_pos_2, imem)
        phi_3 = self.isotropic_moi(charge_pos, elec_pos_3, imem)
        phi_4 = self.isotropic_moi(charge_pos, elec_pos_4, imem)
        return (phi_0 + phi_1 + phi_2 + phi_3 + phi_4)/5

    def potential_at_elec_big_average(self, charge_pos, elec_pos, r, imem=1):
        """ Calculate the potential at electrode 'elec_index' with many points"""
        #import pylab as pl
        number_of_points = 0
        splits = 100
        phi = 0
        for n_x in xrange(splits):
            for n_y in xrange(splits):
                e_x = -r + 2*n_x*r/(splits-1)
                e_y = -r + 2*n_y*r/(splits-1)
                if not np.sqrt(e_x**2 + e_y**2) <= r:
                    continue
                number_of_points += 1
                phi += self.point_source_moi_at_elec(charge_pos, [e_x, e_y, elec_pos[2]], imem)
        return phi/number_of_points


    def find_signal_to_cell(self, cell, mapping):
        """ Calculating the potential at the electrodes,
        given the mapping from the make_mapping method."""

        ntsteps = len(cell.imem[0,:])
        n_elecs = mapping.shape[0]
        n_compartments = cell.imem.shape[0]
        cell.ext_signals = np.zeros((n_elecs, ntsteps))
        for elec in xrange(n_elecs):
            for comp in xrange(n_compartments):
                cell.ext_signals[elec,:] += mapping[elec, comp] * cell.imem[comp,:]        
        return cell



    def make_mapping_to_cell(self, cell, neur_dict, ext_sim_dict):
        """ Make a mapping given two arrays of electrode positions"""
        print '\033[1;35mMaking mapping for %s...\033[1;m' %neur_dict["name"]
 
        if ext_sim_dict['use_line_source']:
            comp_start = np.array([cell.xstart, cell.ystart, cell.zstart])
            comp_end = np.array([cell.xend, cell.yend, cell.zend])
            comp_length = cell.length

        comp_coors = np.array([cell.xmid, cell.ymid, cell.zmid])
        n_compartments = len(comp_coors[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        mapping = np.zeros((n_elecs,n_compartments))
        steps = ext_sim_dict['moi_steps']
        elec_x = ext_sim_dict['elec_x'] # Array
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Scalar    
        for comp in xrange(n_compartments):
            percentage = (comp+1)*100/n_compartments
            stdout.write("\r%d %% complete" % percentage)
            stdout.flush()
            for elec in xrange(n_elecs):
                elec_pos = [elec_x[elec], elec_y[elec], elec_z]
                charge_pos = comp_coors[:,comp]
                if ext_sim_dict['include_elec']:
                    if ext_sim_dict['use_line_source']:
                        if comp == 0:
                            mapping[elec, comp] += self.potential_at_elec(\
                                charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                        else: 
                            mapping[elec, comp] += self.potential_at_elec_line_source(\
                                comp_start[:,comp], comp_end[:,comp],
                                comp_length[comp], elec_pos, ext_sim_dict['elec_radius'])
                    else:
                        mapping[elec, comp] += self.potential_at_elec(\
                            charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                else:
                    if ext_sim_dict['use_line_source']:
                        mapping[elec, comp] += self.line_source_moi(\
                            comp_start, comp_end, comp_length, elec_pos)
                    else:
                        mapping[elec, comp] += self.isotropic_moi(\
                            charge_pos, elec_pos)
        print ''
        return mapping

    def make_mapping(self, neur_dict, ext_sim_dict):
        """ Make a mapping given two arrays of electrode positions"""
        print '\033[1;35mMaking mapping for %s...\033[1;m' %neur_dict["name"]
        coor_folder = os.path.join(ext_sim_dict['neural_input'],\
                                         neur_dict['name'])
        neur_input = os.path.join(coor_folder, 'coor.npy')
        if ext_sim_dict['use_line_source']:
            comp_start = np.load(os.path.join(coor_folder, 'coor_start.npy'))
            comp_end = np.load(os.path.join(coor_folder, 'coor_end.npy'))
            comp_length = np.load(os.path.join(coor_folder, 'length.npy'))

        comp_coors = np.load(neur_input)
        n_compartments = len(comp_coors[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        mapping = np.zeros((n_elecs,n_compartments))
        steps = ext_sim_dict['moi_steps']
        elec_x = ext_sim_dict['elec_x'] # Array
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Scalar    
        for comp in xrange(n_compartments):
            percentage = (comp+1)*100/n_compartments
            stdout.write("\r%d %% complete" % percentage)
            stdout.flush()
            for elec in xrange(n_elecs):
                elec_pos = [elec_x[elec], elec_y[elec], elec_z]
                charge_pos = comp_coors[:,comp]
                if ext_sim_dict['include_elec']:
                    if ext_sim_dict['use_line_source']:
                        if comp == 0:
                            mapping[elec, comp] += self.potential_at_elec(\
                                charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                        else: 
                            mapping[elec, comp] += self.potential_at_elec_line_source(\
                                comp_start[:,comp], comp_end[:,comp],
                                comp_length[comp], elec_pos, ext_sim_dict['elec_radius'])
                    else:
                        mapping[elec, comp] += self.potential_at_elec(\
                            charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                else:
                    if ext_sim_dict['use_line_source']:
                        mapping[elec, comp] += self.line_source_moi(\
                            comp_start, comp_end, comp_length, elec_pos)
                    else:
                        mapping[elec, comp] += self.isotropic_moi(\
                            charge_pos, elec_pos)
        print ''
        try:
            os.mkdir(ext_sim_dict['output_folder'])
            os.mkdir(os.path.join(ext_sim_dict['output_folder'], 'mappings'))
        except OSError:
            pass
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
        try:
            os.mkdir(os.path.join(ext_sim_dict['output_folder'], 'signals'))
        except OSError:
            pass
        np.save(os.path.join(ext_sim_dict['output_folder'], 'signals', \
                                 'signal_%s.npy' %(neur_dict['name'])), signals)           
        return signals
