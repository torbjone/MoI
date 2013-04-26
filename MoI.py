#!/usr/bin/env python
import numpy as np
from sys import stdout
import os
from os.path import join

if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
import pylab as pl

try:
    from ipdb import set_trace
except:
    pass

import pyximport
pyximport.install()
from cython_funcs import *

class MoI:
    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    Set-up:


              SALINE -> sigma_S 

    <----------------------------------------------------> z = + a
    
              TISSUE -> sigma_T


                   o -> charge_pos = [x,y,z]


    <-----------*----------------------------------------> z = -a               
                 \-> elec_pos = [x,y,z] 

                 ELECTRODE GLASS PLATE -> sigma_G 
        

    '''
    def __init__(self,
                 set_up_parameters = {
                     'sigma_G': 0.0, # Below electrode
                     'sigma_T': 0.3, # Tissue
                     'sigma_S': 3.0, # Saline
                     'slice_thickness': 200., # um
                     'steps' : 20,
                      },
                 debug = False
                 ):
        self.set_up_parameters = set_up_parameters

        self.sigma_G = set_up_parameters['sigma_G']
        self.sigma_T = set_up_parameters['sigma_T']
        self.sigma_S = set_up_parameters['sigma_S']
        self._check_for_anisotropy()
        
        self.slice_thickness = set_up_parameters['slice_thickness']
        self.a = self.slice_thickness/2.
        self.steps = set_up_parameters['steps']


    def _anisotropic_saline_scaling(self):
        """ To make formula work in anisotropic case we scale the conductivity of the
        saline to be a scalar k times the tissue conductivity. (Wait 1990)
        """

        ratios = np.array(self.sigma_S) / np.array(self.sigma_T)

        if np.abs(ratios[0] - ratios[2]) <= 1e-15:
            scale_factor = ratios[0]
        elif np.abs(ratios[1] - ratios[2]) <= 1e-15:
            scale_factor = ratios[1]

        sigma_S_scaled = scale_factor * np.array(self.sigma_T)
        sigma_T_net = np.sqrt(self.sigma_T[0] * self.sigma_T[2])
        sigma_S_net = np.sqrt(sigma_S_scaled[0] * sigma_S_scaled[2])

        print "Sigma_T: %s, Sigma_S: %s, Sigma_S_scaled: %s, scale factor: %g" %(
            self.sigma_T, self.sigma_S, sigma_S_scaled, scale_factor)
        self.anis_W = (sigma_T_net - sigma_S_net)/(sigma_T_net + sigma_S_net)

        

    def _check_for_anisotropy(self):
        """ Checks if input conductivities are tensors or scalars
        and sets self.is_anisotropic correspondingly
        """
        sigmas = [self.sigma_G, self.sigma_T, self.sigma_S]
        anisotropy_list = []
        for sigma in sigmas:
            try:
                if len(sigma) == 3:
                    anisotropy_list.append(True)
                else:
                    raise RuntimeError("Conductivity vector but not with size 3")
            except TypeError:
                anisotropy_list.append(False)

        if not len(anisotropy_list) == 3:
            raise RuntimeError("This should not happen?")

        if len(set(anisotropy_list)) != 1:
            raise RuntimeError("Conductivities of different types.")
        if np.all(anisotropy_list):
            self.is_anisotropic = True
        else:
            self.is_anisotropic = False
        if self.is_anisotropic:
            self._anisotropic_saline_scaling()
            
            if not len(self.sigma_G) == len(self.sigma_T) == len(self.sigma_S):
                raise RuntimeError("Conductivities of different dimensions!")
            if (self.sigma_G[0] == self.sigma_G[1] == self.sigma_G[2]) and \
               (self.sigma_T[0] == self.sigma_T[1] == self.sigma_T[2]) and \
               (self.sigma_S[0] == self.sigma_S[1] == self.sigma_S[2]):
                print "Isotropic conductivities can be given as scalars."
                #raise RuntimeError("Isotropic conductivities should be given as scalars!")
            

    def in_domain(self, elec_pos, charge_pos):
        """ Checks if elec_pos and charge_pos is within valid area.
        Otherwise raise exception."""
        if not np.abs(elec_pos[2] - self.a) <= 1e-14 and\
           not np.abs(elec_pos[2] + self.a) <= 1e-14:
            print "Electrode position: ", elec_pos
            raise RuntimeError("Electrode not within valid range.")
        if np.abs(charge_pos[2]) >= self.a:
            print "Charge position: ", charge_pos
            raise RuntimeError("Charge not within valid range.")
        dist = np.sqrt( np.sum( (np.array(charge_pos) - np.array(elec_pos))**2 ))
        if dist < 1e-6:
            print "Charge position: ", charge_pos, "Electrode position: ", elec_pos
            raise RuntimeError("Charge and electrode at same position!")


    ## def ad_hoc_anisotropic(self, charge_pos, elec_pos, imem=1):
    ##     """ Calculate the moi point source potential"""
    ##     self.in_domain(elec_pos, charge_pos)
    ##     x0, y0, z0 = charge_pos[:]
    ##     x, y, z = elec_pos[:]
        
    ##     WTS = (np.min(self.sigma_T) - np.min(self.sigma_S))/\
    ##         (np.min(self.sigma_T) + np.min(self.sigma_S))
        
    ##     def _omega(dz):
    ##         return 1/np.sqrt(self.sigma_T[0]*self.sigma_T[2]*(y - y0)**2 + \
    ##                          self.sigma_T[0]*self.sigma_T[1]*dz**2 + \
    ##                          self.sigma_T[1]*self.sigma_T[2]*(x - x0)**2) 
    ##     phi = _omega(-self.a - z0)
    ##     n = 1
    ##     while n < self.steps:
    ##         phi += WTS**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
    ##         n += 1   
    ##     phi *= 2*imem/(4*np.pi)
    ##     return phi

    def anisotropic_saline_scaling(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi point source potential with saline conductivity
        sigma_S is scaled to k * sigma_T"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]

        def _omega(dz):
            return 1/np.sqrt(self.sigma_T[0]*self.sigma_T[2]*(y - y0)**2 + \
                             self.sigma_T[0]*self.sigma_T[1]*dz**2 + \
                             self.sigma_T[1]*self.sigma_T[2]*(x - x0)**2) 
        phi = _omega(-self.a - z0)
        n = 1
        while n < self.steps:
            phi += (self.anis_W)**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        phi *= 2*imem/(4*np.pi)
        return phi


    def anisotropic_simple(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi point source potential with new WTS"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]

        ratio_variable = self.sigma_S[0]/self.sigma_T[0] +\
                         self.sigma_S[1]/self.sigma_T[1] +\
                         self.sigma_S[2]/self.sigma_T[2]
        
        WTS = (3 - ratio_variable)/(3 + ratio_variable)
        
        def _omega(dz):
            return 1/np.sqrt(self.sigma_T[0]*self.sigma_T[2]*(y - y0)**2 + \
                             self.sigma_T[0]*self.sigma_T[1]*dz**2 + \
                             self.sigma_T[1]*self.sigma_T[2]*(x - x0)**2) 
        phi = _omega(-self.a - z0)
        n = 1
        while n < self.steps:
            phi += WTS**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        return 2*phi*imem/(4*np.pi)




    ## def _anisotropic_moi(self, charge_pos, elec_pos, imem=1):
    ##     """ This function calculates the potential at the position elec_pos = [x,y,z]
    ##     set up by the charge at position charge_pos = [x,y,z]. To get get the potential
    ##     from multiple charges, the contributions must be summed up.
        
    ##     """
    ##     def _MX_function(sigma, x_dist, y_dist, p):
    ##         if sigma[0]*sigma[1]*sigma[2] == 0:
    ##             return 0
    ##         else:
    ##             return sigma[2] * sigma[1] * sigma[2]/\
    ##                    (sigma[1]*sigma[0]*p**2 + \
    ##                     sigma[0]*sigma[2]*y_dist**2 + sigma[2]*sigma[1]*x_dist**2)

    ##     def _WTX(sigma_T, sigma_X, x_dist, y_dist, p):
    ##         M2 = _MX_function(sigma_T, p, y_dist, x_dist)
    ##         MX = _MX_function(sigma_X, p, y_dist, x_dist)
    ##         W_value =  (M2 - MX)/(M2 + MX)
    ##         return W_value

    ##     def _R(sigma, x_dist, y_dist, z_dist):
    ##         return ( sigma[1]*sigma[2] * x_dist**2 +\
    ##                   sigma[0]*sigma[2] * y_dist**2 +\
    ##                   sigma[0]*sigma[1] * z_dist**2)**(-0.5)

    ##     self.in_domain(elec_pos, charge_pos) # Check if valid positions
    ##     factor_lower = 1
    ##     factor_upper = 1
    ##     delta = 1
    ##     x0, y0, z0 = charge_pos[:]
    ##     x, y, z = elec_pos[:]
    ##     n = 1
    ##     phi = _R(self.sigma_T, x - x0, y - y0, z - z0)

    ##     while n < self.steps:
    ##         p_upper = (2*n - 1) * self.a + z0
    ##         p_lower = (2*n - 1) * self.a - z0
    ##         z_dist_lower = (-1)**(n+1)*(-2*n*self.a - z0) - z
    ##         z_dist_upper = (-1)**(n+1)*( 2*n*self.a - z0) - z
    ##         if n%2 == 1: # Odd terms
    ##             factor_lower *= _WTX(self.sigma_T, self.sigma_G, \
    ##                                   x-x0, y-y0, p_lower)
    ##             factor_upper *= _WTX(self.sigma_T, self.sigma_S,\
    ##                                  x-x0, y-y0, p_upper)
    ##         else:
    ##             factor_lower *= _WTX(self.sigma_T, self.sigma_S,\
    ##                                  x-x0, y-y0, p_lower)
    ##             factor_upper *= _WTX(self.sigma_T, self.sigma_G,\
    ##                                 x-x0, y-y0, p_upper)
    ##         delta = factor_upper * _R(self.sigma_T, x-x0, y - y0, z_dist_upper) +\
    ##                 factor_lower * _R(self.sigma_T, x-x0, y - y0, z_dist_lower)
    ##         #print "\n", elec_pos
    ##         #print "p_upper: ", p_upper 
    ##         #print "p_lower: ",p_lower 
    ##         #print "x_dist_lower: ",x_dist_lower
    ##         #print "x_dist_upper: ",x_dist_upper
    ##         #print "factor_lower: ", factor_lower
    ##         #print "factor_upper: ",factor_upper
    ##         #print "phi",phi
    ##         #print "delta",delta
    ##         phi += delta
    ##         n += 1
    ##     phi *= imem/(4*np.pi)
    ##     return phi


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
        WTS = (self.sigma_T - self.sigma_S)/(self.sigma_T + self.sigma_S)
        WTG = (self.sigma_T - self.sigma_G)/(self.sigma_T + self.sigma_G)
        while n < self.steps:
            if n == 0:
                phi += WTS * _omega(z + z0 - (4*n + 2)*self.a) +\
                       WTG * _omega(z + z0 + (4*n + 2)*self.a)
            else:
                phi += (WTS*WTG)**n *(\
                    WTS * _omega(z + z0 - (4*n + 2)*self.a) + WTG * _omega(z + z0 + (4*n + 2)*self.a) +\
                    _omega(z - z0 + 4*n*self.a) + _omega(z - z0 - 4*n*self.a) )
            n += 1
        phi *= imem/(4*np.pi*self.sigma_T)
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
        W = (self.sigma_T - self.sigma_S)/(self.sigma_T + self.sigma_S)
        def _omega(a_z):
            #See Rottman integration formula 46) page 137 for explanation
            factor_a = comp_length*comp_length
            factor_b = - a_x*dx - a_y*dy - a_z * dz
            factor_c = a_x*a_x + a_y*a_y + a_z*a_z
            b_2_ac = factor_b*factor_b - factor_a * factor_c
            if np.abs(b_2_ac) <= 1e-16:
                num = factor_a + factor_b
                den = factor_b
            else:
                num = factor_a + factor_b + \
                      comp_length*np.sqrt(factor_a + 2*factor_b + factor_c)
                den = factor_b + comp_length*np.sqrt(factor_c)
            return np.log(num/den)
        phi = _omega(-self.a - z0)
        if not phi == phi:
            set_trace()
        n = 1
        while n < self.steps:
            phi += W**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        phi *= 2*imem/(4*np.pi*self.sigma_T * comp_length)
        return phi

    def point_source_moi_at_elec(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi point source potential"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        W = (self.sigma_T - self.sigma_S)/(self.sigma_T + self.sigma_S)
        def _omega(dz):
            return 1/np.sqrt( (y - y0)**2 + (x - x0)**2 + dz**2) 
        phi = _omega(-self.a - z0)
        n = 1
        while n < self.steps:
            phi += W**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        return 2*phi*imem/(4*np.pi*self.sigma_T)
    

    ## def potential_at_elec_line_source(self, comp_start, comp_end, comp_length, elec_pos, r, imem=1):
    ##     """ Calculate the potential at electrode 'elec_index' """
    ##     elec_pos_1 = [elec_pos[0], elec_pos[1] + r, elec_pos[2]]
    ##     elec_pos_2 = [elec_pos[0], elec_pos[1] - r, elec_pos[2]]
    ##     elec_pos_3 = [elec_pos[0] + r, elec_pos[1], elec_pos[2]]
    ##     elec_pos_4 = [elec_pos[0] - r, elec_pos[1], elec_pos[2]]
    ##     phi_0 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos, imem)    
    ##     phi_1 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_1, imem)    
    ##     phi_2 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_2, imem)
    ##     phi_3 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_3, imem)
    ##     phi_4 = self.line_source_moi(comp_start, comp_end, comp_length, elec_pos_4, imem)
    ##     return (phi_0 + phi_1 + phi_2 + phi_3 + phi_4)/5

    ## def potential_at_elec(self, charge_pos, elec_pos, r, imem=1):
    ##     """ Calculate the potential at electrode 'elec_index' """
    ##     elec_pos_1 = [elec_pos[0], elec_pos[1] + r, elec_pos[2]]
    ##     elec_pos_2 = [elec_pos[0], elec_pos[1] - r, elec_pos[2]]
    ##     elec_pos_3 = [elec_pos[0] + r, elec_pos[1], elec_pos[2]]
    ##     elec_pos_4 = [elec_pos[0] - r, elec_pos[1], elec_pos[2]]
    ##     phi_0 = self.isotropic_moi(charge_pos, elec_pos, imem)    
    ##     phi_1 = self.isotropic_moi(charge_pos, elec_pos_1, imem)    
    ##     phi_2 = self.isotropic_moi(charge_pos, elec_pos_2, imem)
    ##     phi_3 = self.isotropic_moi(charge_pos, elec_pos_3, imem)
    ##     phi_4 = self.isotropic_moi(charge_pos, elec_pos_4, imem)
    ##     return (phi_0 + phi_1 + phi_2 + phi_3 + phi_4)/5

    def potential_at_elec_big_average(self, elec_pos, r, n_avrg_points, function, func_args):
        """ Calculate the potential at electrode 'elec_index' with n_avrg_points points"""
        phi = 0
        for pt in xrange(n_avrg_points):
            pt_pos = np.array([(np.random.rand() - 0.5)* 2 * r,
                               (np.random.rand() - 0.5)* 2 * r])
            # If outside electrode
            while np.sum( pt_pos**2) > r**2:
                pt_pos = np.array([(np.random.rand() - 0.5) * 2 * r,
                                   (np.random.rand() - 0.5) * 2 * r])
            avrg_point_pos = [elec_pos[0] + pt_pos[0],
                              elec_pos[1] + pt_pos[1],
                              elec_pos[2]]           
            #phi += self.point_source_moi_at_elec(charge_pos, avrg_point_pos, imem)
            phi += function(*func_args, elec_pos=avrg_point_pos)
        return phi/n_avrg_points


    ## def find_signal_to_cell(self, cell, mapping):
    ##     """ Calculating the potential at the electrodes,
    ##     given the mapping from the make_mapping method."""

    ##     ntsteps = len(cell.imem[0,:])
    ##     n_elecs = mapping.shape[0]
    ##     n_compartments = cell.imem.shape[0]
    ##     cell.ext_signals = np.zeros((n_elecs, ntsteps))
    ##     for elec in xrange(n_elecs):
    ##         for comp in xrange(n_compartments):
    ##             cell.ext_signals[elec,:] += mapping[elec, comp] * cell.imem[comp,:]        
    ##     return cell



    ## def make_mapping_to_cell(self, cell, neur_dict, ext_sim_dict):
    ##     """ Make a mapping given two arrays of electrode positions"""
    ##     print '\033[1;35mMaking mapping for %s...\033[1;m' %neur_dict["name"]
 
    ##     if ext_sim_dict['use_line_source']:
    ##         comp_start = np.array([cell.xstart, cell.ystart, cell.zstart])
    ##         comp_end = np.array([cell.xend, cell.yend, cell.zend])
    ##         comp_length = cell.length

    ##     comp_coors = np.array([cell.xmid, cell.ymid, cell.zmid])
    ##     n_compartments = len(comp_coors[0,:])
    ##     n_elecs = ext_sim_dict['n_elecs']
    ##     mapping = np.zeros((n_elecs,n_compartments))
    ##     steps = ext_sim_dict['moi_steps']
    ##     elec_x = ext_sim_dict['elec_x'] # Array
    ##     elec_y = ext_sim_dict['elec_y'] # Array
    ##     elec_z = ext_sim_dict['elec_z'] # Scalar    
    ##     for comp in xrange(n_compartments):
    ##         percentage = (comp+1)*100/n_compartments
    ##         stdout.write("\r%d %% complete" % percentage)
    ##         stdout.flush()
    ##         for elec in xrange(n_elecs):
    ##             elec_pos = [elec_x[elec], elec_y[elec], elec_z]
    ##             charge_pos = comp_coors[:,comp]
    ##             if ext_sim_dict['include_elec']:
    ##                 if ext_sim_dict['use_line_source']:
    ##                     if comp == 0:
    ##                         mapping[elec, comp] += self.potential_at_elec(\
    ##                             charge_pos, elec_pos, ext_sim_dict['elec_radius'])
    ##                     else: 
    ##                         mapping[elec, comp] += self.potential_at_elec_line_source(\
    ##                             comp_start[:,comp], comp_end[:,comp],
    ##                             comp_length[comp], elec_pos, ext_sim_dict['elec_radius'])
    ##                 else:
    ##                     mapping[elec, comp] += self.potential_at_elec(\
    ##                         charge_pos, elec_pos, ext_sim_dict['elec_radius'])
    ##             else:
    ##                 if ext_sim_dict['use_line_source']:
    ##                     mapping[elec, comp] += self.line_source_moi(\
    ##                         comp_start, comp_end, comp_length, elec_pos)
    ##                 else:
    ##                     mapping[elec, comp] += self.isotropic_moi(\
    ##                         charge_pos, elec_pos)
    ##     print ''
    ##     return mapping



    def make_mapping(self, neur_dict, ext_sim_dict):
        """ Make a mapping given two arrays of electrode positions"""
        print '\033[1;35mMaking mapping for %s...\033[1;m' %neur_dict["name"]
        coor_folder = join(ext_sim_dict['neural_input'],\
                                         neur_dict['name'])
        if ext_sim_dict['use_line_source']:
            comp_start = np.load(join(coor_folder, 'coor_start.npy'))
            comp_end = np.load(join(coor_folder, 'coor_end.npy'))
            comp_length = np.load(join(coor_folder, 'length.npy'))
        comp_coors = np.load(join(coor_folder, 'coor.npy'))

        try:
            if ext_sim_dict['collapse_cells']:
                pos = ext_sim_dict['collapse_pos']
                comp_start[2,:] = pos
                comp_end[2,:] = pos
                comp_coors[2,:] = pos
                length = np.sqrt(np.sum((comp_end - comp_start)**2, axis=0))
        except KeyError:
            pass
        n_compartments = len(comp_coors[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        mapping = np.zeros((n_elecs,n_compartments))
        steps = ext_sim_dict['moi_steps']
        elec_x = ext_sim_dict['elec_x'] # Array
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Scalar

        for comp in xrange(n_compartments):
            if os.environ.has_key('DISPLAY'):
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
            if os.environ.has_key('DISPLAY'):                
                print ''
        try:
            os.mkdir(ext_sim_dict['output_folder'])
            os.mkdir(join(ext_sim_dict['output_folder'], 'mappings'))
        except OSError:
            pass
        np.save(join(ext_sim_dict['output_folder'], 'mappings', 'map_%s.npy' \
                %(neur_dict['name'])), mapping)
        return mapping

    def make_mapping_cython(self, ext_sim_dict, xmid=None, ymid=None, zmid=None,
                            xstart=None, ystart=None, zstart=None,
                            xend=None, yend=None, zend=None):
        """ Make a mapping given two arrays of electrode positions"""
        print "Making mapping. Cython style"
        elec_x = ext_sim_dict['elec_x'] # Array
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Scalar    

        if xmid != None:
            xmid = np.array(xmid, order='C')
            ymid = np.array(ymid, order='C')
            zmid = np.array(zmid, order='C')

        if xstart != None:
            xend = np.array(xend, order='C')
            yend = np.array(yend, order='C')
            zend = np.array(zend, order='C')
            xstart = np.array(xstart, order='C')
            ystart = np.array(ystart, order='C')
            zstart = np.array(zstart, order='C')
            

        if ext_sim_dict['include_elec']:
            n_avrg_points = ext_sim_dict['n_avrg_points']
            elec_r = ext_sim_dict['elec_radius']
            if ext_sim_dict['use_line_source']:
                function =  LS_with_elec_mapping
                func_args = [self.sigma_T, self.sigma_S,
                        elec_z, self.steps, n_avrg_points, elec_r,
                        elec_x, elec_y, xstart, ystart, zstart, xend, yend, zend]
            else:
                function = PS_with_elec_mapping
                func_args = [self.sigma_T, self.sigma_S, elec_z,
                        self.steps, n_avrg_points,
                        elec_r, elec_x, elec_y, xmid, ymid, zmid]
        else:
            if ext_sim_dict['use_line_source']:
                function = LS_without_elec_mapping
                func_args = [self.sigma_T, self.sigma_S,
                        elec_z, self.steps, elec_x, elec_y,
                        xstart, ystart, zstart, xend, yend, zend]
            else:
                function = PS_without_elec_mapping
                func_args = [self.sigma_T, self.sigma_S,
                        elec_z, self.steps, elec_x, elec_y, xmid, ymid, zmid]
        mapping = function(*func_args)
        return mapping

    def make_mapping_standalone(self, ext_sim_dict, xmid=None, ymid=None, zmid=None,
                                xstart=None, ystart=None, zstart=None,
                                xend=None, yend=None, zend=None):
        """ Make a mapping given two arrays of electrode positions"""
        print "Making mapping. Python style"
        steps = ext_sim_dict['moi_steps']
        elec_x = ext_sim_dict['elec_x'] # Array
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Scalar    
        
        n_elecs = len(elec_x)
        if ext_sim_dict['include_elec']:
            n_avrg_points = ext_sim_dict['n_avrg_points']

        if ext_sim_dict['use_line_source']:
            function = self.line_source_moi
            n_compartments = len(xstart)
        else:
            function =  self.point_source_moi_at_elec
            n_compartments = len(xmid)
        
        mapping = np.zeros((n_elecs,n_compartments))
        for comp in xrange(n_compartments):
            for elec in xrange(n_elecs):
                elec_pos = [elec_x[elec], elec_y[elec], elec_z]
                if ext_sim_dict['use_line_source']:
                    comp_start = np.array([xstart[comp], ystart[comp], zstart[comp]])
                    comp_end = np.array([xend[comp], yend[comp], zend[comp]])
                    comp_length = np.sqrt(np.sum((comp_end - comp_start)**2))
                    func_args = [comp_start, comp_end, comp_length]
                else:
                    charge_pos = [xmid[comp], ymid[comp], zmid[comp]]
                    func_args = [charge_pos]
                if ext_sim_dict['include_elec']:
                    mapping[elec, comp] += self.potential_at_elec_big_average(\
                        elec_pos, ext_sim_dict['elec_radius'], n_avrg_points, function, func_args)
                else:
                    mapping[elec, comp] += function(*func_args, elec_pos=elec_pos)
        return mapping

    ## def make_mapping_LS_testing(self, ext_sim_dict, xstart, ystart, zstart, xend, yend, zend):
    ##     """ Make a mapping given two arrays of electrode positions"""

    ##     comp_start = np.array([xstart[:], ystart[:], zstart[:]])
    ##     comp_end = np.array([xend[:], yend[:], zend[:]])
    ##     comp_length = np.sqrt(np.sum((comp_start - comp_end)**2, axis=0))
    ##     steps = ext_sim_dict['moi_steps']
    ##     elec_x = ext_sim_dict['elec_x'] # Array
    ##     elec_y = ext_sim_dict['elec_y'] # Array
    ##     elec_z = ext_sim_dict['elec_z'] # Scalar    
    ##     n_compartments = len(comp_start[0,:])
    ##     n_elecs = len(elec_x)
    ##     mapping = np.zeros((n_elecs,n_compartments))
    ##     for comp in xrange(n_compartments):
    ##         for elec in xrange(n_elecs):
    ##             elec_pos = [elec_x[elec], elec_y[elec], elec_z]
    ##             mapping[elec, comp] += self.line_source_moi(\
    ##                 comp_start[:,comp], comp_end[:,comp], comp_length[comp], elec_pos)
    ##     return mapping


    ## def make_mapping_standalone(self, ext_sim_dict):
    ##     """ Make a mapping given two arrays of electrode positions"""

    ##     coor_folder = ext_sim_dict['neural_input']
    ##     if ext_sim_dict['use_line_source']:
    ##         comp_start = np.load(join(coor_folder, 'coor_start.npy'))
    ##         comp_end = np.load(join(coor_folder, 'coor_end.npy'))
    ##         comp_length = np.load(join(coor_folder, 'length.npy'))
    ##     comp_coors = np.load(join(coor_folder, 'coor.npy'))

    ##     steps = ext_sim_dict['moi_steps']
    ##     elec_x = ext_sim_dict['elec_x'] # Array
    ##     elec_y = ext_sim_dict['elec_y'] # Array
    ##     elec_z = ext_sim_dict['elec_z'] # Scalar    
    ##     n_compartments = len(comp_coors[0,:])
    ##     n_elecs = len(elec_x)
    ##     mapping = np.zeros((n_elecs,n_compartments))

    ##     for comp in xrange(n_compartments):
    ##         if os.environ.has_key('DISPLAY'):
    ##             percentage = (comp+1)*100/n_compartments
    ##             stdout.write("\r%d %% complete" % percentage)
    ##             stdout.flush()
    ##         for elec in xrange(n_elecs):
    ##             elec_pos = [elec_x[elec], elec_y[elec], elec_z]
    ##             charge_pos = comp_coors[:,comp]
    ##             if ext_sim_dict['include_elec']:
    ##                 if ext_sim_dict['use_line_source']:
    ##                     if comp == 0:
    ##                         mapping[elec, comp] += self.potential_at_elec(\
    ##                             charge_pos, elec_pos, ext_sim_dict['elec_radius'])
    ##                     else: 
    ##                         mapping[elec, comp] += self.potential_at_elec_line_source(\
    ##                             comp_start[:,comp], comp_end[:,comp],
    ##                             comp_length[comp], elec_pos, ext_sim_dict['elec_radius'])
    ##                 else:
    ##                     mapping[elec, comp] += self.potential_at_elec(\
    ##                         charge_pos, elec_pos, ext_sim_dict['elec_radius'])
    ##             else:
    ##                 if ext_sim_dict['use_line_source']:
    ##                     mapping[elec, comp] += self.line_source_moi(\
    ##                         comp_start, comp_end, comp_length, elec_pos)
    ##                 else:
    ##                     mapping[elec, comp] += self.isotropic_moi(\
    ##                         charge_pos, elec_pos)
    ##     return mapping




    def find_signal_at_electrodes(self, neur_dict, ext_sim_dict, mapping):
        """ Calculating the potential at the electrodes,
        given the mapping from the make_mapping method."""
        
        print '\033[1;35mFinding signal at electrodes from %s ...\033[1;m' % neur_dict['name']
        neur_input = join(ext_sim_dict['neural_input'],
                            neur_dict['name'], 'imem.npy')
        imem =  np.load(neur_input)
        ntsteps = len(imem[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        n_compartments = len(imem[:,0])
        #signals = np.zeros((n_elecs, ntsteps))
        #for elec in xrange(n_elecs):
        #    for comp in xrange(n_compartments):
        #        signals[elec,:] += mapping[elec, comp] * imem[comp,:]
        signals = np.dot(mapping, imem)
        try:
            os.mkdir(join(ext_sim_dict['output_folder'], 'signals'))
        except OSError:
            pass
        np.save(join(ext_sim_dict['output_folder'], 'signals', \
                                 'signal_%s.npy' %(neur_dict['name'])), signals)           
        return signals

    def plot_mea(self, neuron_dict, ext_sim_dict, neural_sim_dict):
        pl.close('all')
        fig_all = pl.figure(figsize=[10,15])
        ax_all = fig_all.add_axes([0.05, 0.03, 0.9, 0.9], frameon=False)
        for elec in xrange(len(ext_sim_dict['elec_x'])):
            ax_all.plot(ext_sim_dict['elec_x'][elec], ext_sim_dict['elec_y'][elec], color='b',\
                    marker='$E%i$'%elec, markersize=20 )    
        legends = []
        for i, neur in enumerate(neuron_dict):
            folder = join(neural_sim_dict['output_folder'], neuron_dict[neur]['name'])
            xmid, ymid, zmid = np.load(folder + '/coor.npy')
            ax_all.plot(xmid[0], ymid[0], marker='$%s$'%neur[-2:], markersize=20)
        ax_all.axis('equal')
        ax_all.set_xlabel('x [mm]')
        ax_all.set_ylabel('y [mm]')
        fig_all.savefig(join(neural_sim_dict['output_folder'], 'all_numbered.png'))
