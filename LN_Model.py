
import numpy as np 
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from utils import calculate_VG_baseline, calculate_n_eq


class LN_Model(object):
    
    def __init__(self,
                 linear_filter,   
                 convolution,
                 nonlinearity,
                 
                 params,

                 convolution_type = 'same',
                 nV = False,
                 stimnorm = False,
                 
                 polarities = ['ON'],
                 filterlength = 1,
                 dt = 0.02):
        

        self.linear_filter = linear_filter      
        self.convolution = convolution
        self.nonlinearity = nonlinearity
        self.stimnorm = stimnorm
        self.params = params

        self.convolution_type = convolution_type
        self.nV = nV

        self.filterlength = filterlength
        self.polarities = polarities
        self.dt = dt
        
        
        


    def predict(self,
                stim,
                time,
                stimulus_name,
                print_steps = True):

        
        '''
        function to make a prediction with the full model 
        returns OPL_inputs as functions ,solutions of the dynamical system and nonlinear_response
        '''


        if self.stimnorm == 'moving':
            print("stimulus moving average")

            stimn = np.zeros(len(stim))
            
            memory = 1.0
            memory_tps = int(memory/self.dt)
            # Loop through the array t o 
            #consider every window of size 3 

            for i in range(len(stim)):
            
                # Calculate the average of current window 
                avg = np.sum(stim[i-memory_tps:i])/memory_tps
            
                # Store the average of current 
                # window in moving average list 
                stimn[i] =stim[i]-avg 

            stim = stimn

        if self.stimnorm == 'mean':
            stim = (stim-stim.mean())#/stim.std()
            print("stimulus mean")

        simt = time[-1]
        tau_B = self.params['tau_B']
        tau_A1 = self.params['tau_A1']
        tau_A2 = self.params['tau_A2']
        tau_VR = self.params['tau_VR']

        # calculate scaling factors to keep intermediate voltage responses at similar amplitudes
        SF_B = self.params['SFB']           # [mV/s**2]
        SF_A1 = self.params['SFA1']              # [mV/s**2]
        SF_A2 = self.params['SFA2']             # [mV/s**2]


        #make timeline for filter with the same dt
        filtertime = np.arange(0,self.filterlength,self.dt)

        if self.convolution_type == 'same': 

            if self.polarities[0] is 'ON':
                filter_B = self.linear_filter(filtertime,tau_B, tau_A2,SF_A2)

            if self.polarities[0] is 'OFF':
                filter_B = -1 * self.linear_filter(filtertime,tau_B, tau_A2,SF_A2)



        if self.convolution_type == 'VR': 

            if self.polarities[0] is 'ON':
                filter_B = self.linear_filter(filtertime,tau_VR)

            if self.polarities[0] is 'OFF':
                filter_B = -1 * self.linear_filter(filtertime,tau_VR)

            
                
        if print_steps == True:

            print("filter computed")

        

        #convolve
        FB = SF_B * self.convolution(stim,np.flip(filter_B),self.dt)

        if print_steps == True:

            print("convolution sucessful")

        
        # prepare OPL Inputs 
        print(len(time))
        print(len(FB))
    


        #apply nonlinearity
        nonlinear_response = np.asarray([self.nonlinearity(l,self.params) for l in FB])


        #some tests 
       
        
        out = { 'name': stimulus_name,
                'stimulus': stim,
                'time': time,
                "sol" : FB,
                "RG": nonlinear_response }


        return out 


    def  plot_kernels(self):


        if self.convolution_type == 'same' : 
            tau_B = self.params['tau_B']
            tau_A1 = self.params['tau_A1']
            tau_A2 = self.params['tau_A2']
            SF_A2 = self.params['SFA2']

        if self.convolution_type == 'VR':
            tau_B = self.params['tau_VR']
            tau_A1 = self.params['tau_VR']
            tau_A2 = self.params['tau_VR']

        #make timeline for filter with the same dt
        filtertime = np.arange(0,self.filterlength,self.dt)

        fig = plt.figure()
        plt.plot(filtertime,self.linear_filter(filtertime,tau_B,tau_A2,SF_A2), color = 'g', label = "kernel B")
        print(self.linear_filter(filtertime,tau_A1,0,0))
        plt.legend()
        plt.title( "Filter shapes for OPL input")
        plt.show()
        return fig





    def plot_response(self,simulation, xlims =(0.9,2)): 

        name = simulation['name']
        stimulus = simulation['stimulus']
        time = simulation['time']
        sol = simulation['sol']
        RG = simulation['RG']

        fig = plt.figure(figsize = (10,5))


        ax0 = fig.add_subplot(311)
        ax0.plot(time,stimulus)

        ax1 = fig.add_subplot(312)
        ax1.plot(time,sol, color = 'g', linestyle = '--',label = 'FB(t)')

        ax2 = fig.add_subplot(313)
        ax2.plot(time,RG, color = 'k', linestyle = '--',label = 'FB(t)')




        ax0.set_title('Stimulus')
        ax0.set_xlim(xlims)

        ax1.set_title('Linear Response')
        ax1.set_xlim(xlims)

        ax2.set_title('Nonlinear Response ')
        ax2.set_xlim(xlims)


    
        fig.legend()
        fig.suptitle(f'{name}')
        plt.show()

        return fig 