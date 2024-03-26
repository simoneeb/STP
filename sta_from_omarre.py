import os
from tqdm.auto import tqdm
import numpy as np
import csv
import h5py
#import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle
import scipy.optimize
from math import pi
import pandas as pd
#import pyret
from matplotlib.backends.backend_pdf import PdfPages
import scipy
import json



sampling_frequency = 20000.0
#framerate = 1/40


#template = f'{which_date}.templates-2.hdf5'
#template = f'/user/sebert/home/Documents/Experiments/Results/{experiment_name}/recording_0/recording_0.templates-merged.hdf5'
#templates = h5py.File(template, 'r')
#grades = templates['tagged'][()].flatten()


#plots_path = f'/user/sebert/home/Documents/Experiments/Results/{experiment_name}/plots/sta/'


STAs = []
spatial = []
temporal = []
polarity = []
nb_spikes = []
rasters_check = []
rasters_check_repeated = []

class Checkerboard:

    def __init__(self, nb_checks, binary_source_path, repetitions, triggers):

        assert os.path.isfile(binary_source_path)

        self._nb_checks = nb_checks
        self._binary_source_path = binary_source_path
        self._repetitions = repetitions
        self._triggers = triggers

        self._binary_source_file = open(self._binary_source_path, mode='rb')


    def __exit__(self, exc_type, exc_value, traceback):

        self._input_file.close()

        return

    def get_limits(self):

        return self._triggers.get_limits()

    def get_repetition_limits(self):

        start_trigger_nbs = self._repetitions.get_start_trigger_nbs(condition_nb=0)
        end_trigger_nbs = self._repetitions.get_end_trigger_nbs(condition_nb=0)

        start_sample_nbs = self._triggers.get_sample_nbs(start_trigger_nbs)
        end_sample_nbs = self._triggers.get_sample_nbs(end_trigger_nbs)

        repetition_limits = [
            (start_sample_nb, end_sample_nb)
            for start_sample_nb, end_sample_nb in zip(start_sample_nbs, end_sample_nbs)
        ]

        return repetition_limits

    def get_image_nbs(self, sample_nbs):

        trigger_nbs = self._triggers.get_trigger_nbs(sample_nbs)

        sequence_length = 300  # frames

        image_nbs = np.copy(trigger_nbs)
        for k, trigger_nb in enumerate(trigger_nbs):
            sequence_nb = trigger_nb // sequence_length
            is_in_frozen_sequence = (sequence_nb % 2) == 1
            if is_in_frozen_sequence:
                offset = 0
            else:
                offset = (sequence_nb // 2) * sequence_length
            image_nb = offset + trigger_nb % sequence_length
            image_nbs[k] = image_nb

        return image_nbs

    def _get_bit(self, bit_nb):

        byte_nb = bit_nb // 8
        self._binary_source_file.seek(byte_nb)
        byte = self._binary_source_file.read(1)
        byte = int.from_bytes(byte, byteorder='big')
        bit = (byte & (1 << (bit_nb % 8))) >> (bit_nb % 8)

        return bit

    def get_image_shape(self):

        shape = (self._nb_checks, self._nb_checks)

        return shape

    def get_image(self, image_nb):

        # # TODO slow, try to read bytes in advance.
        # start_bit_nb = self._nb_checks * self._nb_checks * image_nb
        # start_byte_nb = start_bit_nb // 8
        # end_bit_nb = self._nb_checks * self._nb_checks * (image_nb + 1)
        # end_byte_nb = end_bit_nb // 8
        # nb_bytes = end_byte_nb - start_byte_nb
        #
        # self._input_file.seek(start_byte_nb)
        # bytes = self._input_file.read(nb_bytes)
        # bytes = int.from_bytes(bytes, byteorder='big')

        shape = self.get_image_shape()
        image = np.zeros(shape, dtype=np.float)

        for i in range(0, self._nb_checks):
            for j in range(0, self._nb_checks):
                bit_nb = (self._nb_checks * self._nb_checks * image_nb) + (self._nb_checks * i) + j
                bit = self._get_bit(bit_nb)
                if bit == 0:
                    image[i, j] = 1.0
                elif bit == 1:
                    image[i, j] = 0.0
                else:
                    message = "Unexpected bit value: {}".format(bit)
                    raise ValueError(message)

        image = np.flipud(image)
        image = np.fliplr(image)

        return image

    def get_clip_shape(self, nb_images):

        shape = (nb_images,) + self.get_image_shape()

        return shape

    def get_clip(self, reference_image_nb, nb_images):

        shape = self.get_clip_shape(nb_images)
        clip = np.zeros(shape, dtype=np.float)

        for k in range(0, nb_images):
            image_nb = reference_image_nb + (k - (nb_images - 1))
            clip[k] = self.get_image(image_nb)

        return clip


def align_triggers_spikes(triggers, spike_times, sampling_rate=20000):
    # Clip the spike times to the recording time
    trigger_min = np.min(triggers)
    trigger_max = np.max(triggers)
    clipped_spike_times = clip_list(spike_times, trigger_min, trigger_max)

    # Set trigger start times to zero
    triggers_first_time = np.min(triggers)
    triggers = triggers - triggers_first_time

    # Do the same operation on spike times
    new_spike_times = clipped_spike_times - triggers_first_time
    
    # Get the values in seconds
    new_triggers = triggers/sampling_rate
    new_spike_times = new_spike_times/sampling_rate
    
    return new_triggers, new_spike_times


def clip_list(input_list, min_value, max_value):
    
    clipped_list = input_list[input_list <= max_value]
    clipped_list = clipped_list[clipped_list >= min_value]
    
    return clipped_list

def restrict_array(array, value_min, value_max):
    array = array[array>=value_min]
    array = array[array<=value_max]
    return array.tolist()

def evaluate_polarity(sta):
    """Separate space and time components."""

    time_width, space_height, space_width = sta.shape[0], sta.shape[1], sta.shape[2]
    rf_shape = (time_width, space_height * space_width)
    rf = np.reshape(sta, rf_shape)

    # # Remove the median.
    # rf_median = np.median(rf)
    # rf = rf - rf_median

    u, s, vh = np.linalg.svd(rf, full_matrices=False)

    time_rf_shape = (time_width,)
    time_rf = np.reshape(u[:, 1], time_rf_shape)  # TODO why 1 instead of 0?
    space_rf_shape = (space_height, space_width)
    space_rf = np.reshape(vh[1, :], space_rf_shape)  # TODO understand why 1 instead of 0?

    # Determine the cell polarity
    if np.abs(np.max(rf) - np.median(rf)) >= np.abs(np.min(rf) - np.median(rf)):
        rf_polarity = 'ON'
    else:
        rf_polarity = 'OFF'
        
    return rf_polarity

def separate_components(sta):
    """Separate space and time components."""

    time_width, space_height, space_width = sta.shape[0], sta.shape[1], sta.shape[2]
    rf_shape = (time_width, space_height * space_width)
    rf = np.reshape(sta, rf_shape)

    # # Remove the median.
    # rf_median = np.median(rf)
    # rf = rf - rf_median

    u, s, vh = np.linalg.svd(rf, full_matrices=False)

    time_rf_shape = (time_width,)
    time_rf = np.reshape(u[:, 1], time_rf_shape)  # TODO why 1 instead of 0?
    space_rf_shape = (space_height, space_width)
    space_rf = np.reshape(vh[1, :], space_rf_shape)  # TODO understand why 1 instead of 0?

    # Determine the cell polarity
    if np.abs(np.max(rf) - np.median(rf)) >= np.abs(np.min(rf) - np.median(rf)):
        rf_polarity = 'ON'
    else:
        rf_polarity = 'OFF'
        
    # Determine the spatial RF polarity
    if np.abs(np.max(space_rf) - np.median(space_rf) >= np.abs(np.min(space_rf) - np.median(space_rf))):
        space_rf_polarity = 'ON'
    else:
        space_rf_polarity = 'OFF'
        
    # Determine the temporal RF polarity
    if np.abs(np.max(time_rf) - np.median(time_rf) >= np.abs(np.min(time_rf) - np.median(time_rf))):
        time_rf_polarity = 'ON'
    else:
        time_rf_polarity = 'OFF'
        
    # Reverse components (if necessary).
    if rf_polarity != space_rf_polarity:
        space_rf = - space_rf
        
    if rf_polarity != time_rf_polarity:
        time_rf = - time_rf

    return time_rf, space_rf





def temp_to_unit(key):

    unit = int(key.split('_')[-1])+1

    return 'Unit_{:04d}'.format(unit)



# load stimulus
experiment_name = 'MR-0609'
fpstim  = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/stimuli/{experiment_name}_checker/checkerboard_nd4_stim.mat'

stim =h5py.File(fpstim, 'r')
nb_frames = len(stim['stim'])
nb_checks = stim[stim['stim'][0][0]][()].shape[0]


stimmat = np.zeros((nb_frames, nb_checks,nb_checks))
for i in range(int(nb_frames/2)):
    frame = stim[stim['stim'][i][0]][()]
    stimmat[i,:,:] = frame


# stimmat2 = np.rot90(np.fliplr(stimmat))
stimmat = np.rot90(np.fliplr(stimmat), axes=(1,2),k = 3)

# load binned spikes
fpspikes = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/STA/{experiment_name}/spikes_count_checkerboard_nd4.txt'
spikes = np.loadtxt(fpspikes)
nb_bins = spikes.shape[1]


nb_cells = spikes.shape[0]
framerate = 30.0
dt = 1/framerate
temporal_dimension = 12
time = np.arange(0,nb_bins)*dt
ftime = np.arange(0,temporal_dimension)*dt
ftime = np.flip(ftime)*-1




# choose a cell 
data = {}

for cell in tqdm(range(nb_cells)):
    #cell = 9
    key = f'temp_{cell}'
    unit = temp_to_unit(key)

    
    try:
        fp = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/{experiment_name}/results'
        fpkey = f'{fp}/spiketimes/{unit}_trials.json'
        
        with open(fpkey, 'r', encoding='utf-8') as handle:
            raster = json.load(handle)
          
        spike_count = spikes[cell,:]
        nb_spks = np.sum(spike_count)
    
        sta = sta = np.zeros((temporal_dimension,nb_checks,nb_checks))
    
        # if repetition == 0:
        for frame in range(temporal_dimension,nb_bins):
            clip = stimmat[frame-temporal_dimension:frame,:,:] # why plus 1 ??
            sta += spike_count[frame]*clip  # Add the minus sign because the stimulus is reversed on the DMD TODO change
        #else:
            #   for frame in range(nb_frames_by_repetition):
            #      sequence_offset = 600+repetition*nb_frames_by_repetition
            #     clip = checkerboard_24checks_120000frames[sequence_offset+frame-temporal_dimension+1:sequence_offset+frame+1,:,:]
            #    sta += binned_spikes[repetition,frame]*clip
                #   cumulated_spikes += binned_spikes[repetition,frame]
        if nb_spks > 0:
            sta = sta/nb_spks
    
        #sta = np.flip(sta,axis = 0)
        time_rf, space_rf = separate_components(sta) 
    
        #time_rf = np.flip(time_rf)
        pol = evaluate_polarity(sta)
        # TODO get frame by frame temporal STA
  
    
        # plot and save output
        fpout =f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/STA/{experiment_name}'
        
    
        fig,ax = plt.subplots(1,temporal_dimension, figsize = (20,5))
        fig.subplots_adjust(wspace=1)
        rf_pol = evaluate_polarity(sta)
    
    
        for i in range(temporal_dimension):
            ax[i].imshow(sta[i,:,:])
            ax[i].set_title(f't = -{(temporal_dimension-1-i)*dt:.2f}')
            ax[i].set_xlabel('x')
        ax[0].set_xlabel('y')
    
        #plt.show()
        fig.suptitle(f' STA frames {key}, polarity {pol}')
        fig.savefig(f'{fpout}/plots/full/full_{key}.png')
        plt.close()
    
        fig, ax = plt.subplots(1,2, figsize = (10,5))
        ax[0].imshow(space_rf)
        ax[1].plot(ftime,time_rf)
        ax[0].set_title('spatial')
        ax[1].set_title('temporal')
        ax[1].set_box_aspect(1)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
    
        fig.suptitle(f' STA components {key}, polarity {pol}')
    
        ax[1].set_xlabel('time [s]')
    
        #plt.show()
        fig.savefig(f'{fpout}/plots/components/components_{key}.png')
        plt.close()
        
        
        data[unit] = {}
        data[unit]['sta'] = sta
        data[unit]['space_rf'] = space_rf
        data[unit]['time_rf'] = time_rf
        data[unit]['pol'] = pol
        data[unit]['count'] = spike_count
        data[unit]['nb_spks'] = nb_spks
    
    except:
          print(f'{unit} json file not found')
          continue
   


with open(f'{fpout}/sta_data.pkl', "wb") as handle:   #Pickling
    pickle.dump(data, handle,protocol=pickle.HIGHEST_PROTOCOL )






    
          
       
        
