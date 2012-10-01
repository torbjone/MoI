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
