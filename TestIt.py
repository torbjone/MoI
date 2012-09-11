import random
import unittest
import numpy as np
from MoI import MoI

class TestMoI(unittest.TestCase):

    def test_homogeneous(self):
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

    def test_saline_effect(self):
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

    def test_charge_closer(self):
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
        
    def test_within_domain_check(self):
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
                
if __name__ == '__main__':
    unittest.main()
