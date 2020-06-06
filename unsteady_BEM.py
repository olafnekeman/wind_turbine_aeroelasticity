# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:26:48 2020

@author: TUDelft SID
"""

import numpy as np
import pandas as pd
from steady_BEM import steady_BEM
import matplotlib.pyplot as plt

def pitt_peters(Ct,vind,Uinf,R,dt,glauert=True ):
    # this function determines the time derivative of the induction at the annulli 
    # Ct is the thrust coefficient on the actuator, vind is the induced velocity, 
    # Uinf is the unperturbed velocity and R is the radial scale of the flow, nd dt is the time step
    # it returns the new value of induction vindnew and the time derivative dvind_dt
    # glauert defines if glauert's high load correction is applied
    a=-vind/Uinf # determine the induction coefficient for the time step {i-1}
    Ctn= -CTfunction(a, glauert) # calculate the thrust coefficient from the induction for the time step {i-1}

    dvind_dt =  (Ct-Ctn)/(16/(3*np.pi))*(Uinf**2/R) # calculate the time derivative of the induction velocity
    vindnew = vind + dvind_dt*dt # calculate the induction at time {i} by time integration
    return vindnew, dvind_dt

def CTfunction(a, glauert = False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    CT = np.zeros(np.shape(a))
    CT = 4*a*(1-a)  
    if glauert:
        CT1=1.816;
        a1=1-np.sqrt(CT1)/2;
        CT[a>a1] = CT1-4*(np.sqrt(CT1)-1)*(1-a[a>a1])
    
    return CT
  
    
def ainduction(CT):
    """
    This function calculates the induction factor 'a' as a function of thrust coefficient CT 
    including Glauert's correction
    """
    a = np.zeros(np.shape(CT))
    CT1=1.816;
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a

def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    This function calcualte steh combined tip and root Prandtl correction at agiven radial position 'r_R' (non-dimensioned by rotor radius), 
    given a root and tip radius (also non-dimensioned), a tip speed ratio TSR, the number lf blades NBlades and the axial induction factor
    """
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp1 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot

def loadBladeElement(vnorm, vtan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    calculates the load in the blade element
    """
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm,vtan)
    alpha = twist + inflowangle*180/np.pi
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5*vmag2*cl*chord
    drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(inflowangle)+drag*np.sin(inflowangle)
    ftan = lift*np.sin(inflowangle)-drag*np.cos(inflowangle)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord
    return fnorm , ftan, gamma

def solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist_no_pitch, polar_alpha, polar_cl, polar_cd, time_array, pitch_array, initial_cond ):
    """
    solve balance of momentum between blade element load and loading in the streamtube
    input variables:
    Uinf - wind speed at infinity
    r1_R,r2_R - edges of blade element, in fraction of Radius ;
    rootradius_R, tipradius_R - location of blade root and tip, in fraction of Radius ;
    Radius is the rotor radius
    Omega -rotational velocity
    NBlades - number of blades in rotor
    
    initial_cond = np.array([a_init, ap_init, fnorm_init, ftan_init, gamma_init Ct_init])
    """
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2) #  area streamtube
    r_R = (r1_R+r2_R)/2 # centroide
    # initiatlize variables
    a = initial_cond[0] # axial induction
    aline = initial_cond[1] # tangential induction factor
    Prandtl = 1
    
    Niterations = 100
    Erroriterations =0.0000001 # error limit for iteration rpocess, in absolute value of induction
    
    a_time = np.zeros(len(time_array))
    ap_time = np.zeros(len(time_array))
    fnorm_time = np.zeros(len(time_array))
    ftan_time = np.zeros(len(time_array))
    gamma_time = np.zeros(len(time_array))
    Ct_time = np.zeros(len(time_array))
    
    a_time[0] = initial_cond[0]
    ap_time[0] = initial_cond[1]
    fnorm_time[0] = initial_cond[2]
    ftan_time[0] = initial_cond[3]
    gamma_time[0] = initial_cond[4]    
    Ct_time[0] = initial_cond[5]
    dt = time_array[1]-time_array[0]

    for j in range(1,len(time_array)):
        twist = twist_no_pitch+pitch_array[j]

        for i in range(Niterations):
            # ///////////////////////////////////////////////////////////////////////
            # // this is the block "Calculate velocity and loads at blade element"
            # ///////////////////////////////////////////////////////////////////////
            Urotor = Uinf*(1-a) # axial velocity at rotor
            Utan = (1+aline)*Omega*r_R*Radius # tangential velocity at rotor
            # calculate loads in blade segment in 2D (N/m)
            fnorm, ftan, gamma = loadBladeElement(Urotor, Utan, r_R,chord, twist, polar_alpha, polar_cl, polar_cd)
            load3Daxial =fnorm*Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
            # load3Dtan =loads[1]*Radius*(r2_R-r1_R)*NBlades # 3D force in azimuthal/tangential direction (not used here)

            # ///////////////////////////////////////////////////////////////////////
            # //the block "Calculate velocity and loads at blade element" is done
            # ///////////////////////////////////////////////////////////////////////

            # ///////////////////////////////////////////////////////////////////////
            # // this is the block "Calculate new estimate of axial and azimuthal induction"
            # ///////////////////////////////////////////////////////////////////////
            # // calculate thrust coefficient at the streamtube 
            CT = load3Daxial/(0.5*Area*Uinf**2)

            # calculate new axial induction, accounting for Glauert's correction
            #anew =  ainduction(CT)

            vind,dvind_dt = pitt_peters(np.array([-CT]),np.array([-a_time[j-1]*Uinf]),Uinf,Radius,dt)
            
            anew = -vind[0]/Uinf
#            if j == 5:
#                print(i,a,vind)
            # correct new axial induction with Prandtl's correction
            Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew);
            if (Prandtl < 0.0001): 
                Prandtl = 0.0001 # avoid divide by zero
#            anew = anew/Prandtl # correct estimate of axial induction
            a = 0.0*a+1.0*anew # for improving convergence, weigh current and previous iteration of axial induction
            
            # calculate aximuthal induction
            aline_new = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*Radius)**2)
#            aline_new =aline_new/Prandtl # correct estimate of azimuthal induction with Prandtl's correction
            aline = aline_new
#            aline = aline*0.95+aline_new*0.05
            # ///////////////////////////////////////////////////////////////////////////
            # // end of the block "Calculate new estimate of axial and azimuthal induction"
            # ///////////////////////////////////////////////////////////////////////
            
            #// test convergence of solution, by checking convergence of axial induction
            if (np.abs((a-anew)/a) < Erroriterations): 
                # print("iterations")
                # print(i)
                break

        if i == Niterations-1:
            print('Not converged')
        
        a_time[j] = a
        ap_time[j] = aline
        fnorm_time[j] = fnorm
        ftan_time[j] = ftan
        gamma_time[j] = gamma
        Ct_time[j] = CT
        
    return a_time,ap_time,fnorm_time,ftan_time,gamma_time,Ct_time


class unsteady_BEM:
    def __init__(self, airfoil, TipLocation_R, RootLocation_R, NBlades, Radius, Uinf, TSR, N_blade_sec, spacing = 'cosine'):
        self.airfoil = airfoil
        self.NBlades = NBlades
        self.Radius = Radius
        self.Uinf = Uinf
        self.Omega = TSR*Uinf/Radius
        self.N_blade_sec = N_blade_sec
        self.RootLocation_R = RootLocation_R
        self.TipLocation_R = TipLocation_R
        #create the blade division
        if spacing == 'cosine':
            beta_cosine = np.linspace(0,np.pi,N_blade_sec+1)
            self.r_R_dist = RootLocation_R+(TipLocation_R-RootLocation_R)/2*(1-np.cos(beta_cosine))
        elif spacing == 'uniform':
            self.r_R_dist = np.linspace(RootLocation_R,TipLocation_R,N_blade_sec+1)
        else:
            raise ValueError('Spacing method not recognized, please select either "cosine" or "uniform"')
        
        self.r_R_cent = np.zeros(N_blade_sec)
        for i in range(N_blade_sec):
            self.r_R_cent[i] = (self.r_R_dist[i]+self.r_R_dist[i+1])/2
            
        self.twist_no_pitch = -14*(1-self.r_R_cent)
        self.chord_cent = 3*(1-self.r_R_cent)+1
        
        data1=pd.read_csv(airfoil, header=0,
                            names = ["alfa", "cl", "cd", "cm"],  sep='\s+')
        self.polar_alpha = data1['alfa'][:]
        self.polar_cl = data1['cl'][:]
        self.polar_cd = data1['cd'][:]
        self.B_steady = steady_BEM(airfoil, TipLocation_R, RootLocation_R, NBlades, Radius, Uinf, TSR, N_blade_sec, spacing=spacing)
        
    def get_solution(self, time_vec, CT_time, inflow_model = 'pitt_peters'):
        self.time_vec = time_vec
        self.pitch_time = self.B_steady.find_pitch(CT_time)  
        
        self.inflow_model = inflow_model
        start_CT,start_CP,start_results = self.B_steady.get_solution(self.pitch_time[0])
        
        a_time_res = np.zeros([self.N_blade_sec,len(time_vec)])
        ap_time_res = np.zeros([self.N_blade_sec,len(time_vec)])
        fnorm_time_res = np.zeros([self.N_blade_sec,len(time_vec)])
        ftan_time_res = np.zeros([self.N_blade_sec,len(time_vec)])
        gamma_time_res = np.zeros([self.N_blade_sec,len(time_vec)])
        Ct_time_res = np.zeros([self.N_blade_sec,len(time_vec)])
        
        for i in [0]:#range(self.N_blade_sec):
            initial_cond = np.array([start_results[i,0], start_results[i,1], start_results[i,2], start_results[i,3], start_results[i,4], start_results[i,5]])
            print(initial_cond)
            a_time_res[i,:],ap_time_res[i,:],fnorm_time_res[i,:],ftan_time_res[i,:],gamma_time_res[i,:],Ct_time_res[i,:] = solveStreamtube(self.Uinf, 
                       self.r_R_dist[i], self.r_R_dist[i+1], self.RootLocation_R, self.TipLocation_R , 
                       self.Omega, self.Radius, self.NBlades, self.chord_cent[i], self.twist_no_pitch[i], 
                       self.polar_alpha, self.polar_cl, self.polar_cd, time_vec, self.pitch_time, initial_cond )
             
        return a_time_res,ap_time_res,fnorm_time_res,ftan_time_res,gamma_time_res,Ct_time_res
             
if __name__ == "__main__":
    # define flow conditions
    Uinf = 10 # unperturbed wind speed in m/s
    TSR = 10 # tip speed ratio
    Radius = 50
    N_blade_sec = 30
    NBlades = 3
    
    TipLocation_R =  1
    RootLocation_R =  0.2
    pitch = 2
    
    airfoil = 'DU_polar.txt'
    
    B = unsteady_BEM(airfoil, TipLocation_R, RootLocation_R, NBlades, Radius, Uinf, TSR, N_blade_sec,spacing='cosine')
    
    dt = 0.01
    time_range = np.arange(0,25,dt)
    CT_time = 0.5*np.ones(len(time_range))
    CT_time[time_range>1] = 0.9
   
    
    a_time_res,ap_time_res,fnorm_time_res,ftan_time_res,gamma_time_res,Ct_time_res=B.get_solution(time_range,CT_time)
        
        
    _,_,steady_sol1 = B.B_steady.get_solution(B.pitch_time[0])
    _,_,steady_sol2 = B.B_steady.get_solution(B.pitch_time[-1])
    
    plt.plot(time_range,np.ones(len(time_range))*steady_sol1[0,0])
    plt.plot(time_range,np.ones(len(time_range))*steady_sol2[0,0])
    plt.plot(time_range,a_time_res[0,:])
    
    print(time_range[abs(a_time_res[0,:]-steady_sol2[0,0])/steady_sol2[0,0]<0.01][0])
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        