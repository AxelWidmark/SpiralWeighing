"""
The main class
========================================================
"""

import math
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
pi=math.pi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
numTFThreads = 0
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.compat.v1.disable_eager_execution()




class spiral_weighing():
    
    ''' This is the main class and contains the method of inference,
        including the minimisation algorithm. In this class, any tensorflow
        quantity or function begins with an upper-case letter. Other
        quantities begin with a lower-case letter.
        
        The physical quantities are formulated of units:   
        distance (pc),   
        time (year),
        mass (solar mass).
    
        The free parameters of the method are formulated in terms of
        the tensorflow quantities Vector and Params, which are related
        via sigmoid functions. The reason is that while all the elements
        of the Params quantity are bounded on both sides, the Vector
        elements are unbounded. With this formulation, we avoid
        problems associated with the minimisation algorithm hitting
        hard boundaries in its free parameters.
    
        :param coords:              A 2d array, of dimension (N,2), containing the (Z,W) parameters of the data sample.
                                    This should be provided in units (pc) and (km/s). The velocity is then transformed
                                    to units (pc/yr). The velocity coordinate (W) can be supplied with the Sun's
                                    vertical velocity already corrected for.
        :type coords:               float
        :param zbins:               Vector with the bin-edges of the 2d data histogram's bins in Z
        :type zbins:                float
        :param wbins:               Vector with the bin-edges of the 2d data histogram's bins in w
        :type wbins:                float
        :param num_Gs:              The number of Gaussians in the bulk density Gaussian mixture model
        :type num_Gs:               int
        :param rho_scales:          A list of scale heights for the respective matter density components.
        :type rho_scales:           float
        :param interp_resolution:   The number of interpolations points in height for numerical evaluations of the
                                    vertical oscillation angle.
        :type interp_resolution:    int
        :param mask_params:         List of number that determine the outer mask function M (see paper for details).
        :type mask_params:          float
        :param pixels_smear:        Number of pixels to smear over when producing smeared versions of 2d histograms.
        :type pixels_smear:         float
        :param pixels_smear:        Number of pixels to smear over when producing smeared versions of 2d histograms.
        :type pixels_smear:         float
        :param one_arm:             Specifies one arm, e.g. if a symmetric spiral component is allowed.
        :type one_arm:              bool
        :param mask_z:              Mask in Z, in units of pc.
        :type mask_z:               float
    
        '''
    
    def __init__(self, coords, zbins=np.linspace(-1e3, 1e3, 101), \
            wbins=np.linspace(-60., 60., 121), num_Gs=6, rho_scales=[100., 200., 400., 800.], interp_resolution=1000, \
            mask_params=[[300., 800.], [20., 44.], [2.5e-1, 1e-1]], pixels_smear=2., one_arm=True, mask_z=None):        
        # algorithm is run in units: pc, year, solar mass
        self.kms_over_pcyear = 1.022e-6  # (km/s) / (pc/year)
        self.four_pi_G = 4.*pi*4.49e-15 # in units (pc/year)^2 pc (solar mass)^(-1)
        self.km_over_pc = 3.241e-14
        self.second_over_Myr = 3.171e-14
        self.second_over_yr = 3.171e-8
        self.pcMyr_over_kms = self.km_over_pc/self.second_over_Myr
        self.wbins = self.kms_over_pcyear*wbins
        self.zbins = zbins
        self.zvec = (self.zbins[0:len(self.zbins)-1] + self.zbins[1:len(self.zbins)])/2.
        self.wvec = (self.wbins[0:len(self.wbins)-1] + self.wbins[1:len(self.wbins)])/2.
        self.Grid_spacing = tf.constant([np.abs(self.zbins[-1]-self.zbins[0])/(len(self.zbins)-1), \
            np.abs(self.wbins[-1]-self.wbins[0])/(len(self.wbins)-1)], dtype=tf.float64)
        self.Zero = tf.constant(0., dtype=tf.float64)
        
        self.interp_resolution = interp_resolution
        self.Angle_anchor_points = tf.constant(mask_params[0], dtype=tf.float64)
        
        self.Coords = tf.constant(coords, dtype=tf.float64)
        self.num_stars = len(coords)
        self.Zbins = tf.constant(self.zbins, dtype=tf.float64)
        self.Wbins = tf.constant(self.wbins, dtype=tf.float64)
        self.Zvec = tf.constant(self.zvec, dtype=tf.float64)
        self.Wvec = tf.constant(self.wvec, dtype=tf.float64)
        self.num_Gs = num_Gs
        self.num_sechs = len(rho_scales)
        self.Rho_scales = tf.constant(rho_scales, dtype=tf.float64)
        self.hist2d,xedges,yedges = np.histogram2d(coords[:,0], coords[:,1], bins=[len(self.zbins)-1, len(self.wbins)-1],   \
             range=[[self.zbins[0],self.zbins[-1]],[self.wbins[0],self.wbins[-1]]])
        xx,yy = np.meshgrid(np.linspace(-(len(self.wbins)-1.)/2., (len(self.wbins)-1.)/2., len(self.wbins)), \
            np.linspace(-(len(self.zbins)-1.)/2., (len(self.zbins)-1.)/2., len(self.zbins)))
        self.hist_smear = np.exp(-(xx**2+yy**2)/(2.*pixels_smear**2))
        self.hist_smear /= np.sum(self.hist_smear)
        self.hist2d_convolved = convolve2d(self.hist2d, self.hist_smear, 'same')
        self.hist2d_reldiff = (self.hist2d-self.hist2d_convolved)/self.hist2d_convolved
        self.Hist2d = tf.constant(self.hist2d, dtype=tf.float64)
        self.Hist2d_convolved = tf.constant(self.hist2d_convolved, dtype=tf.float64)
        self.Hist2d_reldiff = tf.constant(self.hist2d_reldiff, dtype=tf.float64)
        zmesh, wmesh = np.meshgrid(self.zbins, self.wbins)
        zmesh, wmesh = (zmesh.T, wmesh.T)
        sigmoid = lambda x: 1./(1.+np.exp(-x))
        self.conditional_grid_inner = np.array( [[sigmoid(      \
            ((z/mask_params[0][0])**2 + (w/mask_params[1][0]/self.kms_over_pcyear)**2 - 1.)/mask_params[2][0]  )    \
            for w in self.wvec] for z in self.zvec] )
        self.conditional_grid_outer = np.array( [[sigmoid(      \
            (1. - (z/mask_params[0][1])**2 - (w/mask_params[1][1]/self.kms_over_pcyear)**2)/mask_params[2][1]  )    \
            for w in self.wvec] for z in self.zvec] )
        if mask_z!=None:
            self.conditional_grid_outer *= np.array( [[(np.abs(z)>np.abs(mask_z)) for w in self.wvec] for z in self.zvec] )
        self.conditional_grid = self.conditional_grid_inner * self.conditional_grid_outer
        self.Conditional_grid_inner = tf.constant(self.conditional_grid_inner, dtype=tf.float64)
        self.Conditional_grid_outer = tf.constant(self.conditional_grid_inner, dtype=tf.float64)
        self.Conditional_grid = tf.constant(self.conditional_grid, dtype=tf.float64)
        self.Z_max = tf.constant(1.5*mask_params[0][1], dtype=tf.float64)
        self.W_max = tf.constant(mask_params[1][1], dtype=tf.float64)
        self.one_arm = one_arm
        return
    
    
    # GRAVITATIONAL POTENTIAL
    @tf.function # return in units of pc^2/yr^2
    def Phi_of_z(self, z, Rho_params, Z_sun):
        ''' Gravitational potential as a function of z.
            
            :param z:           Height with respect to the Sun.
            :type z:            tf.float64
            :param Rho_params:  Mid-plane matter densities.
            :type Rho_params:   tf.float64
            :param Z_sun:       Height of the Sun w.r.t the Galactic mid-plane.
            :type Z_sun:        tf.float64
        
            :return:            Gravitational potential.
            :rtype:             tf.float64
            '''
        Res_km2s2 = self.four_pi_G/self.kms_over_pcyear**2 * \
            tf.reduce_sum( Rho_params[:,None]*tf.math.log(tf.cosh((z[None,:]+Z_sun[None,None])/self.Rho_scales[:,None]))*   \
            tf.pow(self.Rho_scales[:,None], 2), axis=0)
        return self.kms_over_pcyear**2 * Res_km2s2
    
    
    # GRAVITATIONAL ACCELERATION
    @tf.function # return in units of pc/yr^2
    def Acc_of_z(self, z, Rho_params, Z_sun):
        ''' Gravitational acceleration as a function of z.
            
            :param z:           Height with respect to the Sun.
            :type z:            tf.float64
            :param Rho_params:  Mid-plane matter densities.
            :type Rho_params:   tf.float64
            :param Z_sun:       Height of the Sun w.r.t the Galactic mid-plane.
            :type Z_sun:        tf.float64
        
            :return:            Gravitational acceleration.
            :rtype:             tf.float64
            '''
        Res_kms2 = self.four_pi_G/self.kms_over_pcyear**2 * \
            tf.reduce_sum( Rho_params[:,None]*tf.tanh((z[None,:]+Z_sun[None,None])/self.Rho_scales[:,None])*   \
            self.Rho_scales[:,None], axis=0)
        return self.kms_over_pcyear/self.second_over_yr * Res_km2s2
    
    
    @tf.function
    def Time_between_heights(self, z_0, z_1, Z_max, Rho_params):
        ''' Time in years for a stars to travel between two heights.
            
            :param z_0:         Starting height with respect to the mid-plane.
            :type z_0:          tf.float64
            :param z_0:         Final height with respect to the mid-plane.
            :type z_0:          tf.float64
            :param Z_max:       Maximum height of a star given its vertical energy.
            :type Z_max:        tf.float64
            :param Rho_params:  Mid-plane matter densities.
            :type Rho_params:   tf.float64
        
            :return:            Time between heights (yr).
            :rtype:             tf.float64
            '''
        Ez = Phi_of_z(Z_max, Rho_params, self.Zero)
        Z_diff = (z_1-z_0)/self.interp_resolution
        Z_temp_vec = tf.linspace(z_0+Z_diff/2., z_0-Z_diff/2., self.interp_resolution-1)
        Vel_temp_vec = 2. * tf.sqrt( E_z[None,:]-Phi_of_z(Z_temp_vec, Rho_params, self.Zero)[:,None] )
        return Z_diff * tf.sum(Z_temp_vec[:,None]/Vel_temp_vec, axis=0)
    
    
    # make set of params add up to one
    @tf.function
    def Add_to_1(self, zs):
        ''' This function takes a (N-1) parameters and transforms them to
            N parameters that add to unity.
            
            :param zs:      Input parameters of arbitrary length (N-1).
            :type z_0:      tf.float64
        
            :return:        Array of length N.
            :rtype:         tf.float64
            '''
        fac = tf.concat([1-zs, tf.constant([1], tf.float64)], 0)
        zsb = tf.concat([tf.constant([1], tf.float64), zs], 0)
        return tf.math.cumprod(zsb) * fac
    
    
    @tf.function
    def Bulk_density(self, Params_bulk, Z_sun, W_sun):
        ''' This gives the bulk density distribution,
            given the bulk density parameters.
            
            :param Params_bulk: Bulk parameters.
            :type Params_bulk:  tf.float64
            :param Z_sun:       Height of the Sun w.r.t the Galactic mid-plane.
            :type Z_sun:        tf.float64
            :param W_sun:       Vertical velocity of the Sun
            :type W_sun:        tf.float64
        
            :return:            Bulk density in the shape of the 2d data histogram.
            :rtype:             tf.float64
            '''
        Weights = Params_bulk[0:self.num_Gs]
        Std_z = Params_bulk[self.num_Gs:2*self.num_Gs]
        Std_w = Params_bulk[2*self.num_Gs:3*self.num_Gs]
        Bulk_density_grid = tf.reduce_sum( Weights[:,None,None] * \
            tf.math.exp(-tf.pow((self.Zvec[None,:,None]+Z_sun[None,None,None])/Std_z[:,None,None], 2)/2.)   \
            / tf.math.sqrt(2.*pi*tf.pow(Std_z[:,None,None], 2)) * \
            tf.math.exp(-tf.pow((self.Wvec[None,None,:]+W_sun[None,None,None])/Std_w[:,None,None], 2)/2.)   \
            / tf.math.sqrt(2.*pi*tf.pow(Std_w[:,None,None], 2)), axis=0) \
            * tf.reduce_prod(self.Grid_spacing, axis=0)
        return Bulk_density_grid
    
    # RELATIVE DENSITY DUE TO SPIRAL
    @tf.function
    def Spiral_rel_density(self, Params_spiral):
        ''' This gives the relative phase space perturbation
            of the phase-space spiral,
            given the spiral density parameters.
            
            :param Params_spiral: Spiral parameters.
            :type Params_spiral:  tf.float64
        
            :return:            Spiral density in the shape of the 2d data histogram.
            :rtype:             tf.float64
            '''
        # these are the free parameters that affect the spiral
        Rho_params = Params_spiral[0:self.num_sechs]
        Z_sun = Params_spiral[self.num_sechs] # in units pc
        W_sun = Params_spiral[self.num_sechs+1] # in units pc/yr
        Laps = Params_spiral[self.num_sechs+2] # in units yrs
        Phi_init = Params_spiral[self.num_sechs+3] # the initial phase, one period = 1
        Amplitude = Params_spiral[self.num_sechs+4] # between 0 and 1
        if self.one_arm:
            Single_or_double_arms = 0.
        else:
            Single_or_double_arms = Params_spiral[self.num_sechs+5] # between 0 and 1
        
        
        Zeros_grid = tf.zeros(tf.shape(self.Hist2d), dtype=tf.float64)
        Z_grid = self.Zvec[:,None]+Z_sun[None,None]+Zeros_grid
        W_grid = self.Wvec[None,:]+W_sun[None,None]+Zeros_grid
        Abs_Z_grid = tf.abs(Z_grid)
        Abs_W_grid = tf.abs(Z_grid)
        Ez_grid = self.kms_over_pcyear**2*2/37.*tf.reduce_sum( Rho_params[:,None,None]* \
            tf.math.log(tf.cosh(Z_grid[None,:,:]/self.Rho_scales[:,None,None]))*   \
            tf.pow(self.Rho_scales[:,None,None], 2), axis=0) + tf.pow(W_grid, 2)/2.
        
        Z_diff = self.Z_max/(self.interp_resolution-1)
        # the vector over Z is evenly spaced in the first dimension
        Z_vec_grid = tf.linspace(Zeros_grid, self.Z_max[None,None]+Zeros_grid, self.interp_resolution, axis=0)
        Phi_vec_grid = self.kms_over_pcyear**2*2/37.*tf.reduce_sum( Rho_params[:,None,None,None]* \
                    tf.math.log(tf.cosh(Z_vec_grid[None,:,:,:]/self.Rho_scales[:,None,None,None]))*   \
                    tf.pow(self.Rho_scales[:,None,None,None], 2), axis=0)
        Acc_vec_grid = self.kms_over_pcyear**2*2/37.*tf.reduce_sum( Rho_params[:,None,None,None]* \
                    tf.tanh(Z_vec_grid[None,:,:,:]/self.Rho_scales[:,None,None,None])*   \
                    self.Rho_scales[:,None,None,None], axis=0)
        # the vector over W is NOT evenly spaced in the first dimension
        W_vec_grid = tf.sqrt( 2. * tf.clip_by_value( \
            Ez_grid[None,:,:]-Phi_vec_grid,  \
            self.Zero, tf.constant(1e10, dtype=tf.float64)) )
        Wmid_vec_grid = tf.concat([(W_vec_grid[1:,:,:]+W_vec_grid[:-1,:,:])/2., \
            tf.zeros([1, tf.shape(self.Hist2d)[0], tf.shape(self.Hist2d)[1]], dtype=tf.float64)], axis=0)
        Time_vec_grid_A = tf.clip_by_value(tf.math.divide_no_nan(Z_diff, Wmid_vec_grid), self.Zero, tf.constant(1e9, dtype=tf.float64))
        Wdiff_vec_grid = tf.concat([W_vec_grid[:-1,:,:]-W_vec_grid[1:,:,:], \
            tf.zeros([1, tf.shape(self.Hist2d)[0], tf.shape(self.Hist2d)[1]], dtype=tf.float64)], axis=0)
        Time_vec_grid_B = tf.clip_by_value(tf.math.divide_no_nan(Wdiff_vec_grid, Acc_vec_grid), self.Zero, tf.constant(1e9, dtype=tf.float64))
        
        KineticEfrac_vec_grid = tf.pow(W_vec_grid, 2)/2. / Ez_grid[None,:,:]
        Time_vec_grid = Time_vec_grid_A * tf.math.sigmoid((KineticEfrac_vec_grid-0.75)/0.05) + \
                        Time_vec_grid_B * tf.math.sigmoid((0.75-KineticEfrac_vec_grid)/0.05)
        
        Z_vec_temp = tf.linspace(self.Zero, self.Z_max, self.interp_resolution, axis=0)
        Time_grid = tf.math.reduce_sum(Time_vec_grid *  \
            tf.math.sigmoid(Abs_Z_grid[None,:,:]-Z_vec_temp[:,None,None]+Z_diff[None,None,None]/2.), axis=0)
        Period_grid = 4.*tf.math.reduce_sum(Time_vec_grid, axis=0)
        Zsign_grid = tf.math.sign(tf.math.sign(Z_grid)+0.5)
        Wsign_grid = tf.math.sign(tf.math.sign(W_grid)+0.5)
        Angle_grid = 0.5 - 0.25*(Zsign_grid*Wsign_grid+Zsign_grid) + \
            Zsign_grid*Wsign_grid * Time_grid / Period_grid
        
        
        # compute anchor, which Phi_init and T_pert relative to
        Z_vec_anchor_A = tf.linspace(self.Angle_anchor_points[0]*0.5/self.interp_resolution,    \
            self.Angle_anchor_points[0]*(self.interp_resolution-0.5)/self.interp_resolution, self.interp_resolution-1, axis=0)
        W_vec_anchor_A = tf.sqrt(2.*(self.Phi_of_z(tf.stack([self.Angle_anchor_points[0]]),     \
            Rho_params, self.Zero)[0,None]-self.Phi_of_z(Z_vec_anchor_A, Rho_params, self.Zero)))
        Period_anchor_A = 4.*tf.math.reduce_sum(self.Angle_anchor_points[0]/self.interp_resolution/W_vec_anchor_A, axis=0)
        Z_vec_anchor_B = tf.linspace(self.Angle_anchor_points[1]*0.5/self.interp_resolution,    \
            self.Angle_anchor_points[1]*(self.interp_resolution-0.5)/self.interp_resolution, self.interp_resolution-1, axis=0)
        W_vec_anchor_B = tf.sqrt(2.*(self.Phi_of_z(tf.stack([self.Angle_anchor_points[1]]), Rho_params, self.Zero)[0,None]  \
            -self.Phi_of_z(Z_vec_anchor_B, Rho_params, self.Zero)))
        Period_anchor_B = 4.*tf.math.reduce_sum(self.Angle_anchor_points[1]/self.interp_resolution/W_vec_anchor_B, axis=0)
        Ez_anchors = self.Phi_of_z(self.Angle_anchor_points, Rho_params, self.Zero)
        
        # compute the relative z-vz density of the spiral
        Ez_condition_grid = tf.sigmoid( (Ez_grid-Ez_anchors[0])     \
            / (Ez_anchors[0]-self.Phi_of_z(tf.stack([self.Angle_anchor_points[0]-20.]), Rho_params, self.Zero)[0]) )
        T_pert = tf.abs( Laps / (1./Period_anchor_A - 1./Period_anchor_B) )
        Phi_grid = Phi_init + (tf.math.divide_no_nan(T_pert, Period_grid) - tf.math.divide_no_nan(T_pert, Period_anchor_A))
        Rel_density_grid = Ez_condition_grid * Amplitude * ( (1.-Single_or_double_arms) * tf.cos(2.*pi*(Angle_grid-Phi_grid)) + \
                (Single_or_double_arms) * tf.cos(4.*pi*(Angle_grid-Phi_grid)) )
        
        return Rel_density_grid, Period_grid, Ez_grid
    

    # LIKELIHOOD USING ONLY RELATIVE STELLAR DENSITIES
    @tf.function
    def Log_likelihood_bulk(self, Params):
        ''' Returns the logarithm of the likelihood,
            when fitting only the bulk.
            
            :param Params:  Vector of free parameters.
            :type Params:   tf.float64
        
            :return:        Log likelihood value
            :rtype:             tf.float64
            '''
        Params_spiral = Params[0:self.num_sechs+6]
        Params_bulk = Params[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs]
        Z_sun = Params_spiral[self.num_sechs] # in units pc
        W_sun = Params_spiral[self.num_sechs+1] # in units pc/yr
        Bulk_density_grid = self.Bulk_density(Params_bulk, Z_sun, W_sun)
        #Chi2_grid = 2.*tf.math.log(tf.cosh(Bulk_density_grid-self.Hist2d))
        Chi2_grid = tf.pow(Bulk_density_grid-self.Hist2d, 2)/Bulk_density_grid
        #Chi2 = tf.reduce_sum( Chi2_grid )
        Chi2 = tf.reduce_sum( Chi2_grid * self.Conditional_grid_outer )
        return -Chi2
    
    
    # LIKELIHOOD USING ONLY RELATIVE STELLAR DENSITIES
    @tf.function
    def Log_likelihood_full(self, Params):
        ''' Returns the logarithm of the full likelihood.
            
            :param Params:  Vector of free parameters.
            :type Params:   tf.float64
        
            :return:        Log likelihood value
            :rtype:         tf.float64
            '''
        Params_spiral = Params[0:self.num_sechs+6]
        Params_bulk = Params[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs]
        Z_sun = Params_spiral[self.num_sechs] # in units pc
        W_sun = Params_spiral[self.num_sechs+1] # in units pc/yr
        Rel_density_grid, Period_grid, Ez_grid = self.Spiral_rel_density(Params_spiral)
        Bulk_density_grid = self.Bulk_density(Params_bulk, Z_sun, W_sun)
        Total_density_grid = Bulk_density_grid * (1. + Rel_density_grid)
        Chi2_grid = tf.pow(Total_density_grid-self.Hist2d, 2)/Total_density_grid
        Chi2 = tf.reduce_sum( Chi2_grid * self.Conditional_grid_outer )
        return -Chi2
    
    
    def get_grids(self, vector=None):
        ''' Returns 2d grids with the following quantity: relative density of the spiral,
            time period, vertical energy.
            Output is in the form of numpy arrays (not tensorflow objects).
            
            :param vector: Vector of free parameters.
            :type vector:  float
        
            :return:        Three 2d arrays (rel dens. of spiral, time period, vertical energy)
            :rtype:         float
            '''
        if vector is None:
            vector = self.randomize_vector()
        Vector = tf.Variable(vector, dtype=tf.float64)
        Params_spiral = self.Vector_2_params_spiral(Vector)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            res = sess.run( self.Spiral_rel_density(Params_spiral) )
        return res
    
    
    def randomize_vector(self, full=False):
        ''' Returns a random but reasonable guess for the vector
            of free parameters.
            
            :param full: If true, vector includes bulk density parameters.
            :type full:  bool
            
            :return:        Vector of parameters
            :rtype:         float
            '''
        if full:
            res = np.concatenate( [np.random.normal(loc=-2., scale=.3, size=self.num_sechs), \
                np.random.normal(loc=0., scale=.5, size=6), np.random.normal(loc=0., scale=1.0, size=3*self.num_Gs)] )
        else:
            np.concatenate( [np.random.normal(loc=-2., scale=.3, size=self.num_sechs), \
                            np.random.normal(loc=0., scale=.5, size=6)] )
        return res
    
    
    def get_grids_full(self, vector=None):
        ''' Returns 2d grids with the following quantity: relative density of the spiral,
            time period, vertical energy.
            Output is in the form of numpy arrays (not tensorflow objects).
            
            :param vector: Vector of free parameters.
            :type vector:  float
        
            :return:        Three 2d arrays (rel dens. of spiral, time period, vertical energy)
            :rtype:         float
            '''
        if vector is None:
            vector = self.randomize_vector(full=True)
        Vector = tf.Variable(vector, dtype=tf.float64)
        Params = self.Vector_2_params_full(Vector)
        Params_spiral = Params[0:self.num_sechs+6]
        Params_bulk = Params[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs]
        Z_sun = Params_spiral[self.num_sechs] # in units pc
        W_sun = Params_spiral[self.num_sechs+1] # in units pc/yr
        Rel_density_grid, Period_grid, Ez_grid = self.Spiral_rel_density(Params_spiral)
        Bulk_density_grid = self.Bulk_density(Params_bulk, Z_sun, W_sun)
        Total_density_grid = Bulk_density_grid * (1. + self.Conditional_grid*Rel_density_grid)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            res = sess.run( [Rel_density_grid, Period_grid, Ez_grid, Bulk_density_grid, Total_density_grid] )
        return res
    
    
    def get_potential(self, vector=None):
        ''' Gravitational potential as a function of z,
            returned as numpy vectors.

            :param vector: Vector of free parameters.
            :type vector:  float
        
            :return:        Vector in height (z) and gravitational potential values.
            :rtype:         float
            '''
        Vector = tf.Variable(vector, dtype=tf.float64)
        Params_spiral = self.Vector_2_params_spiral(Vector)
        Rho_params = Params_spiral[0:self.num_sechs]
        Z_sun = Params_spiral[self.num_sechs] # in units pc
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            zvec, potvec = sess.run( [self.Zvec, self.Phi_of_z(self.Zvec, Rho_params, Z_sun)] )
        return zvec, potvec
    
    
    @tf.function
    def Vector_2_params_spiral(self, Vector):
        ''' Transforms a Vector to spiral parameters via the sigmoid function.
            See class description for a more detailed explanation.

            :param vector: Vector of free parameters.
            :type vector:  tf.float64
        
            :return:        Free spiral parameters.
            :rtype:         tf.float64
            '''
        Prior_range = tf.constant(np.concatenate((self.num_sechs*[[0., 0.2]], [[-50., 50.], \
            [-20.*self.kms_over_pcyear, 20.*self.kms_over_pcyear], [0., 4.], [0., 2.], [0., 1.], [0., 1.]])), dtype=tf.float64)
        Params_spiral = Prior_range[:,0] + (Prior_range[:,1]-Prior_range[:,0]) * tf.sigmoid(Vector[0:self.num_sechs+6])
        return Params_spiral
        
    @tf.function
    def Vector_2_params_full(self, Vector):
        ''' Transforms a Vector to the free parameters via the sigmoid function.
            See class description for a more detailed explanation.

            :param vector: Vector of free parameters.
            :type vector:  tf.float64
        
            :return:        Free parameters.
            :rtype:         tf.float64
            '''
        Prior_range = tf.constant(np.concatenate((self.num_sechs*[[0., 0.2]], [[-50., 50.], \
            [-20.*self.kms_over_pcyear, 20.*self.kms_over_pcyear], [0., 4.], [-4., 4.], [0., 1.], [0., 1.]])), dtype=tf.float64)
        Params_spiral = Prior_range[:,0] + (Prior_range[:,1]-Prior_range[:,0]) * tf.sigmoid(Vector[0:self.num_sechs+6])
        Vector_bulk = Vector[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs]
        Weights = 2.*self.num_stars*tf.sigmoid(Vector_bulk[0]) * self.Add_to_1(tf.sigmoid(Vector_bulk[1:self.num_Gs]))
        Std_z = tf.math.cumsum(500.*tf.sigmoid(Vector_bulk[self.num_Gs:2*self.num_Gs]))
        Std_w = self.kms_over_pcyear*200.*tf.sigmoid(Vector_bulk[2*self.num_Gs:3*self.num_Gs])
        Params = tf.concat([Params_spiral, Weights, Std_z, Std_w], axis=0)
        return Params
    
    
    # minimize
    def minimize_spiral_likelihood(self, p0=None, number_of_iterations=1000, print_gap=20, numTFTthreads=0, learning_rate=1e-2, fixed_sun=True):
        ''' Minimises the likelihood when only varying the spiral parameters.

            :param p0:                      Initial guess for the vector.
            :type p0:                       float
            :param number_of_iterations:    Number of iterations of the minimisation algorithm.
            :type number_of_iterations:     int
            :param print_gap:               Number of iterations between result printouts.
            :type print_gap:                int
            :param numTFTthreads:           Number of tensorflow threads. If zero, tensorflow decides itself.
            :type numTFTthreads:            int
            :param learning_rate:           Learning rate of the AdamOptimizer.
            :type learning_rate:            float
            :param fixed_sun:               Whether or not the height of the Sun is fixed or allowed to be a free parameter.
            :type fixed_sun:                bool
        
            :return:        Array of log likelihood values, and array of vectors.
            :rtype:         float
            '''
        if p0 is None:
            vector = self.randomize_vector()
        else:
            vector = p0
        if fixed_sun:
            VectorA = tf.Variable(vector[0:self.num_sechs], dtype=tf.float64)
            VectorB = tf.constant(vector[self.num_sechs:self.num_sechs+2], dtype=tf.float64)
            if self.one_arm:
                VectorC = tf.concat([tf.Variable(vector[self.num_sechs+2:self.num_sechs+5], dtype=tf.float64), \
                    tf.constant([0.], dtype=tf.float64)], axis=0)
            else:
                VectorC = tf.Variable(vector[self.num_sechs+2:self.num_sechs+6], dtype=tf.float64)
            VectorD = tf.constant(vector[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs], dtype=tf.float64)
            Vector = tf.concat([VectorA, VectorB, VectorC, VectorD], axis=0)
        else:
            if self.one_arm:
                VectorA = tf.concat([tf.Variable(vector[0:self.num_sechs+5], dtype=tf.float64), \
                    tf.constant([0.], dtype=tf.float64)], axis=0)
            else:
                VectorA = tf.Variable(vector[0:self.num_sechs+6], dtype=tf.float64)
            VectorB = tf.constant(vector[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs], dtype=tf.float64)
            Vector = tf.concat([VectorA, VectorB], axis=0)
        Params = self.Vector_2_params_full(Vector)
        MinusLogPosterior = -self.Log_likelihood_full(Params)
        Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(MinusLogPosterior)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            for i in range(number_of_iterations):
                _, minusLogPosterior, vector, params  =  \
                    sess.run([Optimizer, MinusLogPosterior, Vector, Params])
                if np.isnan(minusLogPosterior):
                    print(minusLogPosterior)
                    print(list(vector))
                    print(list(params), "\n\n")
                    raise ValueError("AdamOptimizer returned NaN")
                if i%print_gap==0:
                    print(minusLogPosterior)
                    print(list(vector))
                    print(list(params), "\n\n")
        return minusLogPosterior, vector
    
    
    # minimize
    def minimize_full_likelihood(self, p0=None, number_of_iterations=20000, print_gap=20, numTFTthreads=0, learning_rate=1e-2):
        ''' Minimises the full likelihood, i.e. both bulk and spiral are free.

            :param p0:                      Initial guess for the vector.
            :type p0:                       float
            :param number_of_iterations:    Number of iterations of the minimisation algorithm.
            :type number_of_iterations:     int
            :param print_gap:               Number of iterations between result printouts.
            :type print_gap:                int
            :param numTFTthreads:           Number of tensorflow threads. If zero, tensorflow decides itself.
            :type numTFTthreads:            int
            :param learning_rate:           Learning rate of the AdamOptimizer.
            :type learning_rate:            float
        
            :return:        Array of log likelihood values, and array of vectors.
            :rtype:         float
            '''
        if p0 is None:
            vector = self.randomize_vector(full=True)
        else:
            vector = p0
        Vector = tf.Variable(vector, dtype=tf.float64)
        Params = self.Vector_2_params_full(Vector)
        MinusLogPosterior = -self.Log_likelihood_full(Params)
        Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(MinusLogPosterior)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            for i in range(number_of_iterations):
                _, minusLogPosterior, vector, params  =  \
                    sess.run([Optimizer, MinusLogPosterior, Vector, Params])
                if np.isnan(minusLogPosterior):
                    print(minusLogPosterior)
                    print(list(vector))
                    print(list(params), "\n\n")
                    raise ValueError("AdamOptimizer returned NaN")
                if i%print_gap==0:
                    print(minusLogPosterior)
                    print(list(vector))
                    print(list(params), "\n\n")
        return minusLogPosterior, vector
    
    
    # minimize
    def minimize_bulk_likelihood(self, p0=None, number_of_iterations=20000, print_gap=20,   \
            numTFTthreads=0, learning_rate=1e-2, fixed_solar_params=None):
        ''' Minimises the likelihood when only varying the bulk density parameters.

            :param p0:                      Initial guess for the vector.
            :type p0:                       float
            :param number_of_iterations:    Number of iterations of the minimisation algorithm.
            :type number_of_iterations:     int
            :param print_gap:               Number of iterations between result printouts.
            :type print_gap:                int
            :param numTFTthreads:           Number of tensorflow threads. If zero, tensorflow decides itself.
            :type numTFTthreads:            int
            :param learning_rate:           Learning rate of the AdamOptimizer.
            :type learning_rate:            float
            :param fixed_solar_params:      Tuple containing solar height and solar velocity.
            :type fixed_solar_params:       float
            
        
            :return:        Array of log likelihood values, and array of vectors.
            :rtype:         float
            '''
        if p0 is None:
            vector = self.randomize_vector(full=True)
        else:
            vector = p0
        VectorA = tf.constant(vector[0:self.num_sechs], dtype=tf.float64)
        if fixed_solar_params==None:
            VectorB = tf.Variable(vector[self.num_sechs:self.num_sechs+2], dtype=tf.float64)
        else:
            VectorB = tf.constant(fixed_solar_params, dtype=tf.float64)
        VectorC = tf.constant(vector[self.num_sechs+2:self.num_sechs+6], dtype=tf.float64)
        VectorD = tf.Variable(vector[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs], dtype=tf.float64)
        Vector = tf.concat([VectorA, VectorB, VectorC, VectorD], axis=0)
        Params = self.Vector_2_params_full(Vector)
        MinusLogPosterior = -self.Log_likelihood_bulk(Params)
        Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(MinusLogPosterior)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            for i in range(number_of_iterations):
                _, minusLogPosterior, vector, params  =  \
                    sess.run([Optimizer, MinusLogPosterior, Vector, Params])
                if np.isnan(minusLogPosterior):
                    print(minusLogPosterior)
                    print(list(vector))
                    print(list(params), "\n\n")
                    raise ValueError("AdamOptimizer returned NaN")
                if i%print_gap==0:
                    print(minusLogPosterior)
                    print(list(vector))
                    print(list(params), "\n\n")
        return minusLogPosterior, vector
    
    
    #def spiral_hessian(self, vector):
    #    VectorA = tf.Variable(vector[0:self.num_sechs], dtype=tf.float64)
    #    VectorB = tf.constant(vector[self.num_sechs:self.num_sechs+2], dtype=tf.float64)
    #    VectorC = tf.Variable(vector[self.num_sechs+2:self.num_sechs+6], dtype=tf.float64)
    #    VectorD = tf.constant(vector[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs], dtype=tf.float64)
    #    Vector = tf.concat([VectorA, VectorB, VectorC, VectorD], axis=0)
    #    #SubVector = tf.concat([VectorA, VectorC], axis=0)
    #    Params_spiral = self.Vector_2_params_spiral(Vector)
    #    Params = self.Vector_2_params_full(Vector)
    #    MinusLogPosterior = -self.Log_likelihood_full(tf.concat([Params_spiral,     \
    #        Params[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs]], axis=0))
    #    #Hessian = tf.hessians(MinusLogPosterior, Params_spiral)
    #    #print('just grad.')
    #    #Hessian = tf.gradients(MinusLogPosterior, Params_spiral)
    #    print('grad. form.')
    #    Hessian = tf.gradients( tf.gradients(MinusLogPosterior, Params_spiral), Params_spiral )
    #    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
    #    with tf.compat.v1.Session(config=session_conf) as sess:
    #        sess.run( tf.compat.v1.global_variables_initializer() )
    #        hessian, minusLogPosterior = sess.run( [Hessian, MinusLogPosterior] )
    #    return hessian, minusLogPosterior
    
    
    ## takes a chain and returns a stepsize vector
    #def adjust_stepsize(self, samples):
    #    res = 1e-1 * np.std(samples, axis=0)
    #    if np.sum(res)==0.:
    #        res = self.get_stepsize_guess()
    #    return res
    
    
    # get time
    def get_time(self, vector):
        ''' Returns the time since the beginning of the perturbation that gave rise to the spiral.

            :param vector:  Vector.
            :type vector:   float
            
        
            :return:        Time in years.
            :rtype:         float
            '''
        Vector = tf.constant(vector, dtype=tf.float64)
        Params = self.Vector_2_params_full(Vector)
        Rho_params = Params[0:self.num_sechs]
        Laps = Params[self.num_sechs+2]
        # compute anchor, which Phi_init and T_pert relative to
        Z_vec_anchor_A = tf.linspace(self.Angle_anchor_points[0]*0.5/self.interp_resolution,    \
            self.Angle_anchor_points[0]*(self.interp_resolution-0.5)/self.interp_resolution, self.interp_resolution-1, axis=0)
        W_vec_anchor_A = tf.sqrt(2.*(self.Phi_of_z(tf.stack([self.Angle_anchor_points[0]]),     \
            Rho_params, self.Zero)[0,None]-self.Phi_of_z(Z_vec_anchor_A, Rho_params, self.Zero)))
        Period_anchor_A = 4.*tf.math.reduce_sum(self.Angle_anchor_points[0]/self.interp_resolution/W_vec_anchor_A, axis=0)
        Z_vec_anchor_B = tf.linspace(self.Angle_anchor_points[1]*0.5/self.interp_resolution,    \
            self.Angle_anchor_points[1]*(self.interp_resolution-0.5)/self.interp_resolution, self.interp_resolution-1, axis=0)
        W_vec_anchor_B = tf.sqrt(2.*(self.Phi_of_z(tf.stack([self.Angle_anchor_points[1]]),     \
            Rho_params, self.Zero)[0,None]-self.Phi_of_z(Z_vec_anchor_B, Rho_params, self.Zero)))
        Period_anchor_B = 4.*tf.math.reduce_sum(self.Angle_anchor_points[1]/self.interp_resolution/W_vec_anchor_B, axis=0)
        Ez_anchors = self.Phi_of_z(self.Angle_anchor_points, Rho_params, self.Zero)
        T_pert = tf.abs( Laps / (1./Period_anchor_A - 1./Period_anchor_B) )
        with tf.compat.v1.Session() as sess:
            t_pert = sess.run(T_pert)
        return t_pert
    
    ## run MCMC chain
    #def run_HMC_spiral(self, p0=None, steps=1e4, burnin_steps=0, num_adaptation_steps=0, num_leapfrog_steps=3, \
    #            step_size_start=None, num_steps_between_results=0, fixed_sun=True):
    #    if p0 is None:
    #        vector = self.randomize_vector()
    #    else:
    #        vector = p0
    #    VectorB = tf.constant(vector[self.num_sechs:self.num_sechs+2], dtype=tf.float64)
    #    VectorD = tf.constant(vector[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs], dtype=tf.float64)
    #    sub_vector = np.concatenate([vector[0:self.num_sechs], vector[self.num_sechs+2:self.num_sechs+6]])
    #    @tf.function
    #    def the_function(Params):
    #        VectorA = Params[0:self.num_sechs]
    #        VectorC = Params[self.num_sechs:self.num_sechs+4]
    #        Vector = tf.concat([VectorA, VectorB, VectorC, VectorD], axis=0)
    #        Prim_params = self.Vector_2_params_full(Vector)
    #        return self.Log_likelihood_full(Prim_params)
    #    num_results = int(steps)
    #    num_burnin_steps = int(burnin_steps)
    #    adaptive_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
    #            target_log_prob_fn=the_function,
    #            num_leapfrog_steps=num_leapfrog_steps,
    #            step_size=step_size_start),
    #        num_adaptation_steps=int(num_adaptation_steps))
    #    @tf.function
    #    def run_chain():
    #        # Run the chain (with burn-in).
    #        samples, [is_accepted, step_size, log_prob] = tfp.mcmc.sample_chain(
    #            num_results=num_results,
    #            num_burnin_steps=num_burnin_steps,
    #            current_state=sub_vector,
    #            kernel=adaptive_hmc,
    #            num_steps_between_results=num_steps_between_results,
    #            trace_fn=lambda _, pkr: [pkr.inner_results.is_accepted,     \
    #                pkr.inner_results.accepted_results.step_size,   \
    #                pkr.inner_results.accepted_results.target_log_prob])
    #        is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float64))
    #        return samples, is_accepted, step_size, log_prob
    #    Samples, Is_accepted, Step_size, Log_prob = run_chain()
    #    with tf.compat.v1.Session() as sess:
    #        samples, is_accepted, step_size, log_prob = sess.run([Samples, Is_accepted, Step_size, Log_prob])
    #    sssamples = []
    #    for ss in samples:
    #        sssamples.append( np.concatenate( (ss[0:self.num_sechs], vector[self.num_sechs:self.num_sechs+2], \
    #             ss[self.num_sechs:self.num_sechs+4], vector[self.num_sechs+6:self.num_sechs+6+3*self.num_Gs]) ) )
    #    return np.array(sssamples), log_prob, step_size, is_accepted
