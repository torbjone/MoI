import random
import unittest
import numpy as np
from MoI import MoI
import pylab as pl
class TestMoI(unittest.TestCase):

    def test_homogeneous(self):
        """If saline and tissue has same conductivity, MoI formula
        should return 2*(inf homogeneous point source)."""
        set_up_parameters = {
            'sigma_1': 0.0, # Below electrode
            'sigma_2': 0.3, # Tissue
            'sigma_3': 0.3, # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos = [0,0,0]
        elec_pos = [-set_up_parameters['slice_thickness']/2, 0, 0]
        dist = np.sqrt( np.sum(np.array(charge_pos) - np.array(elec_pos))**2)
        expected_ans = 2/(4*np.pi*set_up_parameters['sigma_2'])\
                       * imem/(dist)
        returned_ans = Moi.isotropic_moi(charge_pos, elec_pos, imem)
        self.assertAlmostEqual(expected_ans, returned_ans, 6)

    def test_saline_effect(self):
        """ If saline conductivity is bigger than tissue conductivity, the
        value of 2*(inf homogeneous point source) should be bigger
        than value returned from MoI"""
        set_up_parameters = {
            'sigma_1': 0.0, # Below electrode
            'sigma_2': 0.3, # Tissue
            'sigma_3': 3.0, # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos = [0,0,0]
        elec_pos = [-set_up_parameters['slice_thickness']/2, 0, 0]
        dist = np.sqrt( np.sum((np.array(charge_pos) - np.array(elec_pos))**2))
        expected_ans = 2/(4*np.pi*set_up_parameters['sigma_2']) * imem/dist
        returned_ans = Moi.isotropic_moi(charge_pos, elec_pos, imem)
        self.assertGreater(expected_ans, returned_ans)

    def test_charge_closer(self):
        """ If charge is closer to electrode, the potential should
        be greater"""
        set_up_parameters = {
            'sigma_1': 0.0, # Below electrode
            'sigma_2': 0.3, # Tissue
            'sigma_3': 3.0, # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos_1 = [0,0,0]
        charge_pos_2 = [-set_up_parameters['slice_thickness']/4, 0,0]
        elec_pos = [-set_up_parameters['slice_thickness']/2, 0, 0]
        returned_ans_1 = Moi.isotropic_moi(charge_pos_1, elec_pos, imem)
        returned_ans_2 = Moi.isotropic_moi(charge_pos_2, elec_pos, imem)
        self.assertGreater(returned_ans_2, returned_ans_1)
        
    def test_within_domain_check(self):
        """ Test if unvalid electrode or charge position raises RuntimeError.
        """
        set_up_parameters = {
            'sigma_1': 0.0, # Below electrode
            'sigma_2': 0.3, # Tissue
            'sigma_3': 3.0, # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        a = set_up_parameters['slice_thickness']/2
        invalid_positions = [[-a - 0.12,0,0],
                            [+a + 0.12,0,0]]
        valid_position = [0,0,0]
        kwargs = {'imem': imem}
        with self.assertRaises(RuntimeError):
            Moi.isotropic_moi(valid_position, valid_position)
        for pos in invalid_positions:
            with self.assertRaises(RuntimeError):
                Moi.isotropic_moi(valid_position, pos)
                Moi.isotropic_moi(pos, valid_position)


    def test_very_anisotropic(self):
        """ Made to find error in very anisotropic case close to upper layer
        """
        set_up_parameters = {
            'sigma_1': [1.0, 1.0, 1.0], # Below electrode
            'sigma_2': [0.1, 0.1, 1.0], # Tissue
            'sigma_3': [0.0, 0.0, 0.0], # Saline
            'slice_thickness': 0.2,
            'steps' : 2}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        a = set_up_parameters['slice_thickness']/2.
        high_position = [.09, 0, 0]
        low_position = [-a + 0.01, 0, 0]
        z_array = np.linspace(-0.2,0.2,41)
        y_array = np.linspace(-0.1,0.1,21)
        values_high = []
        values_low = []
        for y in y_array:
            for z in z_array:
                values_high.append([z,y, Moi.anisotropic_moi(\
                    charge_pos = high_position, elec_pos = [-0.1,y,z])])
                values_low.append([z,y, Moi.anisotropic_moi(\
                    charge_pos = low_position, elec_pos = [-0.1,y,z])])
        values_high = np.array(values_high)
        values_low = np.array(values_low)
        pl.subplot(211)
        pl.scatter(values_high[:,0], values_high[:,1], c = values_high[:,2])
        pl.axis('equal')
        pl.colorbar()
        pl.subplot(212)
        pl.scatter(values_low[:,0], values_low[:,1], c = values_low[:,2])
        pl.colorbar()
        pl.axis('equal')
        pl.show()


    def test_big_average(self):
        """ Testing average over electrode with many values"""
        set_up_parameters = {
            'sigma_1': 0.0, # Below electrode
            'sigma_2': 0.3, # Tissue
            'sigma_3': 3.0, # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        a = set_up_parameters['slice_thickness']/2.
        Moi = MoI(set_up_parameters = set_up_parameters)
        r = 0.03
        charge_pos = [0, 0, 0]
        elec_pos = [-a, 0,0]
        phi = Moi.potential_at_elec_big_average(charge_pos, elec_pos, r)


    def atest_moi_line_source(self):
        """ Testing infinite isotropic moi line source formula"""
        set_up_parameters = {
            'sigma_1': 0.0, # Below electrode
            'sigma_2': 0.3, # Tissue
            'sigma_3': 3.0, # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        a = set_up_parameters['slice_thickness']/2.
        Moi = MoI(set_up_parameters = set_up_parameters)

        comp_start = [.09, -.1, -.05]
        comp_end = [-0.09, .1, .01]
        comp_mid = (np.array(comp_end) + np.array(comp_start))/2
        comp_length = np.sqrt( np.sum((np.array(comp_end) - np.array(comp_start))**2))
        elec_y = np.linspace(-0.15, 0.15, 5)
        elec_z = np.linspace(-0.15, 0.15, 5)
        phi_LS = []
        phi_PS = []
        phi_PSi = []
        y = []
        z = []
        points = 3
        s = np.array(comp_end) - np.array(comp_start)
        ds = s / (points)
        
        for z_pos in xrange(len(elec_z)):
            for y_pos in xrange(len(elec_y)):
                phi_LS.append(Moi.line_source_moi(comp_start, comp_end, \
                                           comp_length, [-0.1, elec_y[y_pos], elec_z[z_pos]], imem=1))
                phi_PS.append(Moi.isotropic_moi(comp_mid, [-0.1, elec_y[y_pos], elec_z[z_pos]]))
                delta = 0
                for step in xrange(points+1):
                    pos = comp_start + ds*step
                    delta += Moi.isotropic_moi(\
                        pos, [-0.1, elec_y[y_pos], elec_z[z_pos]], imem = 1./(points+1))
                phi_PSi.append(delta)
                                       
                z.append(elec_z[z_pos])
                y.append(elec_y[y_pos])
        import pylab as pl
        pl.subplot(411)
        pl.scatter(z,y, c=phi_LS, s=2, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()
        pl.subplot(412)
        pl.scatter(z,y, c=phi_PS, s=2, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()
        pl.subplot(413)
        pl.scatter(z,y, c=phi_PSi, s=2, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()

        
        pl.subplot(414)       
        pl.scatter(z,y, c=(np.array(phi_LS) - np.array(phi_PSi)), s=1, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()       
        pl.savefig('line_source_test2.png')
        
if __name__ == '__main__':
    unittest.main()
