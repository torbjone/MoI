import random
import unittest
import numpy as np
from MoI import MoI
import pylab as pl
class TestMoI(unittest.TestCase):

    def atest_homogeneous(self):
        """If saline and tissue has same conductivity, MoI formula
        should return 2*(inf homogeneous point source)."""
        set_up_parameters = {
            'sigma_1': [0.0, 0.0, 0.0], # Below electrode
            'sigma_2': [0.3, 0.3, 0.3], # Tissue
            'sigma_3': [0.3, 0.3, 0.3], # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos = [0,0,0]
        elec_pos = [-set_up_parameters['slice_thickness']/2, 0, 0]
        dist = np.sqrt( np.sum(np.array(charge_pos) - np.array(elec_pos))**2)
            
        expected_ans = 2/(4*np.pi*set_up_parameters['sigma_2'][0])\
                       * imem/(dist)
        returned_ans = Moi.anisotropic_moi(charge_pos, elec_pos, imem)
        self.assertAlmostEqual(expected_ans, returned_ans, 6)

    def atest_saline_effect(self):
        """ If saline conductivity is bigger than tissue conductivity, the
        value of 2*(inf homogeneous point source) should be bigger
        than value returned from MoI"""
        set_up_parameters = {
            'sigma_1': [0.0, 0.0, 0.0], # Below electrode
            'sigma_2': [0.3, 0.3, 0.3], # Tissue
            'sigma_3': [3.0, 3.0, 3.0], # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos = [0,0,0]
        elec_pos = [-set_up_parameters['slice_thickness']/2, 0, 0]
        dist = np.sqrt( np.sum(np.array(charge_pos) - np.array(elec_pos))**2)
            
        expected_ans = 2/(4*np.pi*set_up_parameters['sigma_2'][0])\
                       * imem/(dist)
        returned_ans = Moi.anisotropic_moi(charge_pos, elec_pos, imem)
        self.assertGreater(expected_ans, returned_ans)

    def atest_charge_closer(self):
        """ If charge is closer to electrode, the potential should
        be greater"""
        set_up_parameters = {
            'sigma_1': [0.0, 0.0, 0.0], # Below electrode
            'sigma_2': [0.3, 0.3, 0.3], # Tissue
            'sigma_3': [3.0, 3.0, 3.0], # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos_1 = [0,0,0]
        charge_pos_2 = [-set_up_parameters['slice_thickness']/4, 0,0]
        elec_pos = [-set_up_parameters['slice_thickness']/2, 0, 0]
        returned_ans_1 = Moi.anisotropic_moi(charge_pos_1, elec_pos, imem)
        returned_ans_2 = Moi.anisotropic_moi(charge_pos_2, elec_pos, imem)
        self.assertGreater(returned_ans_2, returned_ans_1)
        
    def atest_within_domain_check(self):
        """ Test if unvalid electrode or charge position raises RuntimeError.
        """
        set_up_parameters = {
            'sigma_1': [0.0, 0.0, 0.0], # Below electrode
            'sigma_2': [0.3, 0.3, 0.3], # Tissue
            'sigma_3': [3.0, 3.0, 3.0], # Saline
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
            Moi.anisotropic_moi(valid_position, valid_position)
        for pos in invalid_positions:
            with self.assertRaises(RuntimeError):
                Moi.anisotropic_moi(valid_position, pos)
                Moi.anisotropic_moi(pos, valid_position)


    def atest_very_anisotropic(self):
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
            'sigma_1': [0.0, 0.0, 0.0], # Below electrode
            'sigma_2': [0.3, 0.3, 0.3], # Tissue
            'sigma_3': [3.0, 3.0, 3.0], # Saline
            'slice_thickness': 0.2,
            'steps' : 20}
        a = set_up_parameters['slice_thickness']/2.
        Moi = MoI(set_up_parameters = set_up_parameters)
        r = 0.03
        charge_pos = [0, 0, 0]
        elec_pos = [-a, 0,0]
        phi = Moi.potential_at_elec_big_average(charge_pos, elec_pos, r)
                
if __name__ == '__main__':
    unittest.main()
