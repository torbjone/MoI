    
    Note that this approach has been implemented in LFPy2.0, and for most purposes we recommend using this software
    https://github.com/LFPy/LFPy
    
    ----
    
    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    Set-up:


              SALINE -> sigma_3 = [sigma_3x, sigma_3y, sigma_3z]

    <----------------------------------------------------> z = + a
    
              TISSUE -> sigma_2 = [sigma_2x, sigma_2y, sigma_2z]


                   o -> charge_pos = [x,y,z]


    <-----------*----------------------------------------> z = -a               
                 \-> elec_pos = [x,y,z] 

                 ELECTRODE -> sigma_3 = [sigma_3x, sigma_3y, sigma_3z]
        


    '''
