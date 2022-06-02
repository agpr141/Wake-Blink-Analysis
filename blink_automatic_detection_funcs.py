"""
Functions script for analysing wake blink eye movements from EEG&EOG data
author: @agpr141
date completed: 02.06.22

main function 'blink_analyse':
1. extracts & cleans fp1/fp2/eog1/eog2 data from .edf
2. detects blink events
3. extracts blink characteristics such as # blinks, blink rate, blink density, up & down phase duration & gradients,
difference in time between left & right eye onset
4. saves characteristics in .csv file and returns main output dataframe
"""
# import modules
import mne
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy
import pandas
from collections import defaultdict

# define functions used in main 'blink_analyse function


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_id_min(first_list, second_list):
    # Finding the max, and then it's index in the first list
    min_value = min(first_list)
    min_index = first_list.index(min_value)
    # Return the corresponding value in the second list
    return second_list[min_index]


def get_id_max(first_list, second_list):
    # Finding the max, and then it's index in the first list
    max_value = max(first_list)
    max_index = first_list.index(max_value)
    # Return the corresponding value in the second list
    return second_list[max_index]


def slope(x1, y1, x2, y2):
    grad_by_sample = (y2 - y1) / (x2 - x1)
    grad_by_sf = (y2 - y1) / ((x2 / 512) - (x1 / 512))
    return grad_by_sample, grad_by_sf


def blink_analyse(weo, pptid, night, group):
    raw = mne.io.read_raw_edf(weo,
                              preload=True, exclude=['MK', 'LArm', 'RArm', 'LeftArm', 'RightArm', 'C4', 'F4',
                                                     'O2', 'F3', 'C3', 'P3', 'P4', 'O1',
                                                     'T7',
                                                     'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
                                                     'Oz',
                                                     'FT9', 'FT10', 'POz', 'F1', 'F2', 'F5',
                                                     'F6', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5',
                                                     'FC6', 'FT7', 'FT8', 'C1', 'C2', 'C5',
                                                     'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5',
                                                     'CP6', 'TP7', 'TP8', 'P1', 'P2', 'P5',
                                                     'P6', 'PO7', 'PO8', 'ECG', 'EMG1', 'EMG2',
                                                     'EMG3', 'AF7', 'AF8', 'F7',
                                                     'F8', 'M1', 'M2'], verbose=0)  # load in edf

    # re-reference eye channels to Fpz as per BrainProducts instructions
    raw_1 = mne.set_bipolar_reference(inst=raw, anode='E1', cathode='Fpz', ch_name='EOG1', drop_refs=False, copy=False)
    raw_2 = mne.set_bipolar_reference(inst=raw_1, anode='E2', cathode='Fpz', ch_name='EOG2', drop_refs=True, copy=False)
    signal = raw_2
    signal.drop_channels('E1')

    fp1 = signal._data[0]  # isolate fp1 data from mne raw structure & clean
    fp1_cleaned = nk.eog_clean(fp1, sampling_rate=512, method='agarwal2019')  # 'clean' i.e. filter data
    fp1_uv = fp1_cleaned * 1000000  # convert signal from V to uV

    fp2 = signal._data[1]  # isolate fp2 data from mne raw structure & clean
    fp2_cleaned = nk.eog_clean(fp2, sampling_rate=512, method='agarwal2019')  # 'clean' i.e. filter data
    fp2_uv = fp2_cleaned * 1000000  # convert signal from V to uV

    e1 = signal._data[2]  # isolate e1 data from mne raw structure & clean
    e1_cleaned = nk.eog_clean(e1, sampling_rate=512, method='agarwal2019')  # 'clean' i.e. filter data
    invert_e1 = e1_cleaned * -1  # invert signal for peak detection
    e1_uv = invert_e1 * 1000000  # convert signal from V to uV

    e2 = signal._data[3]  # isolate e1 data from mne raw structure & clean
    e2_cleaned = nk.eog_clean(e2, sampling_rate=512, method='agarwal2019')  # 'clean' i.e. filter data
    invert_e2 = e2_cleaned * -1  # invert signal for peak detection
    e2_uv = invert_e2 * 1000000  # convert signal from V to uV

    # peak detect on all 4 signals (extracts blink# and peak-to-peak)
    blinks_fp1 = nk.eog_findpeaks(fp1_uv, sampling_rate=512, threshold=0.35, method="nk", show=False)
    blinks_fp2 = nk.eog_findpeaks(fp2_uv, sampling_rate=512, threshold=0.35, method="nk", show=False)
    blinks_e1 = nk.eog_findpeaks(e1_uv, sampling_rate=512, threshold=0.35, method="nk", show=False)
    blinks_e2 = nk.eog_findpeaks(e2_uv, sampling_rate=512, threshold=0.35, method="nk", show=False)

    # plot to check
#    fig, ax = plt.subplots()
#    ax.plot(fp1_cleaned, linewidth=0.8, color='navy', label='Fp1')
#    ax.plot(fp2_cleaned, linewidth=0.8, color='slateblue', label='Fp2')
#    ax.plot(invert_e1, linewidth=0.8, color='darkorange', label='E1')
#    ax.plot(invert_e2, linewidth=0.8, color='red', label='E2')
#    ax.plot(blinks_fp1, fp1_cleaned[blinks_fp1], 'o', color='darkred', label='Fp1 Peaks')
#    ax.plot(blinks_fp2, fp2_cleaned[blinks_fp2], 'x', color='darkred', label='Fp2 Peaks')
#    ax.plot(blinks_e1, invert_e1[blinks_e1], 'o', color='blue', label='E1 Peaks')
#    ax.plot(blinks_e2, invert_e2[blinks_e2], 'x', color='blue', label='E2 Peaks')
#    plt.legend()

    # remove peaks without all 4 matching
    # compare peak/trough indexes between eyes. keep those in agreement(w/i 30 (17ms) samples of each other)
    # use fp1 as the main comparator
    fp1_peaks = []
    fp2_peaks = []
    e1_peaks = []
    e2_peaks = []

    for index in blinks_fp1:
        fp2_match = find_nearest(blinks_fp2, index)
        e1_match = find_nearest(blinks_e1, index)
        e2_match = find_nearest(blinks_e2, index)
        idx_plus = index + 30
        idx_minus = index - 30
        if idx_minus <= fp2_match <= idx_plus and idx_minus <= e1_match <= idx_plus \
                and idx_minus <= e2_match <= idx_plus:
            fp1_peaks.append(index)
            fp2_peaks.append(fp2_match)
            e1_peaks.append(e1_match)
            e2_peaks.append(e2_match)

    fig, ax = plt.subplots()
    ax.plot(fp1_uv, linewidth=0.8, color='navy', label='Fp1')
    ax.plot(fp2_uv, linewidth=0.8, color='slateblue', label='Fp2')
    ax.plot(e1_uv, linewidth=0.8, color='darkorange', label='E1')
    ax.plot(e2_uv, linewidth=0.8, color='red', label='E2')
    ax.plot(fp1_peaks, fp1_uv[fp1_peaks], 'o', color='darkred', label='Fp1 Peaks')
    ax.plot(fp2_peaks, fp2_uv[fp2_peaks], 'x', color='darkred', label='Fp2 Peaks')
    ax.plot(e1_peaks, e1_uv[e1_peaks], 'o', color='blue', label='E1 Peaks')
    ax.plot(e2_peaks, e1_uv[e2_peaks], 'x', color='blue', label='E2 Peaks')
    plt.legend()

    # populate variables for use in for loop
    chans = [fp1_uv, fp2_uv]
    ch_peaks = [fp1_peaks, fp2_peaks]
    ch_name = ['Fp1', 'Fp2']
    start_list = []
    # create dict struct to store output values in
    outer_dict = defaultdict(dict)
    # for loop to extract blink characteristics
    for (l, m, n) in zip(chans, ch_peaks, ch_name):

        # total blink number
        total_blink = len(m)

        # blink rate
        secs = len(l)/512
        mins = secs/60
        blink_rate = total_blink/mins

        # blink density
        ptp = np.diff(m)  # calculate peak->peak difference between blinks (in samples)
        ptp_s = ptp / 512  # convert to seconds
        blink_density = np.mean(ptp_s)  # calculate mean peak->peak difference in seconds

        # epoch to create blink epoch
        events = nk.epochs_create(l, m, sampling_rate=512, epochs_start=-0.5, epochs_end=0.5)
        events_a = nk.epochs_to_array(events)  # Convert to 2D array
        data_t = events_a.T

        # intitalise variables
        up_phase = []
        down_phase = []
        blink_dur = []
        half_base_dur = []
        blink_amplitude = []
        start_indexes = []
        end_indexes = []
        up_grad = []
        down_grad = []

        counter = -1
        for nest in data_t:
            counter += 1

            try:
                # calculate epoch velocity
                velocity = [sum(nest[:i]) for i in range(len(nest))]
                detrend = scipy.signal.detrend(velocity)
                vel_troughy = np.where((detrend[1:-1] < detrend[0:-2]) * (detrend[1:-1] < detrend[2:]))[
                                 0] + 1  # finds all troughs - should only detect 1
                if vel_troughy.size > 1:  # if more than 1 trough detected, minimum value will be true trough
                    vt_value = []
                    for i in vel_troughy:
                        vt_value.append(detrend[i])
                    velocity_trough = get_id_min(vt_value, vel_troughy)
                else:
                    velocity_trough = int(vel_troughy)
                vel_peaky = np.where((detrend[1:-1] > detrend[0:-2]) * (detrend[1:-1] > detrend[2:]))[
                                 0] + 1  # finds all peaks - might detect 1 or 2
                if vel_peaky.size > 1:  # if more than 1 peak is detected, 2nd will be true blink end
                    vp_value = []
                    for i in vel_peaky:
                        vp_value.append(detrend[i])
                    velocity_peak = get_id_max(vp_value, vel_peaky)
                else:
                    velocity_peak = int(vel_peaky)

                # code to plot velocity detections
                #plt.plot(detrend)
                #plt.plot(velocity_trough, detrend[velocity_trough], 'o', color='darkorange', label='blink starts')
                #plt.plot(velocity_peak, detrend[velocity_peak], 'o', color='mediumvioletred', label='blink ends')

                # calculate 'eyes closing' phase - e.g. from VELOCITY TROUGH (START) TO PEAK
                rev = nest[::-1]  # reverse data to calculate rise
                reverse = rev
                peak_peak_diff = 256 - velocity_trough
                rev_peak_idx = 256 + peak_peak_diff  # max idx
                rev_peak_val = reverse[256]  # max val
                # crop array from max idx to end
                reverse_crop = reverse[rev_peak_idx:]
                # find closest small number to start of cropped data segment
                rev_trough_idx = \
                    np.where((reverse_crop[1:-1] < reverse_crop[0:-2]) * (reverse_crop[1:-1] < reverse_crop[2:]))[
                        0] + 1
                if rev_trough_idx.size == 0:  # for if lowest value is at last value of the cropped data segment
                    rev_trough_idx = len(reverse_crop)
                    rev_trough_idx = np.array([rev_trough_idx])
                    rev_trough_val = reverse_crop[rev_trough_idx-1]
                    rev_trough_val = np.array([rev_trough_val])
                else:
                    rev_trough_idx = rev_trough_idx[0] + rev_peak_idx
                    rev_trough_val = reverse[rev_trough_idx]
                blink_starty = 512-rev_trough_idx
                blink_start_true = 256-blink_starty
                blink_start = int(blink_start_true)
                start_indexes.append(blink_start)

                # identify velocity peak location & crop data onward
                peak_idx = 256
                # calculate down phase  - e.g. from VELOCITY PEAK TO END
                nest_crop = nest[velocity_peak:]  # identify index of peak & crop from that point on
                # find closest small number to start of cropped data segment
                nest_troughs = scipy.signal.find_peaks(-nest_crop)
                tr_listy = list(nest_troughs[0])

                if len(tr_listy) == 0:  # for if lowest value is at last value of the cropped data segment
                    trough_idx = len(nest_crop)
                    trough_idx = np.array([trough_idx])
                    blink_endy = trough_idx
                    blink_end = trough_idx
                else:
                    trough_idx = tr_listy[0] + velocity_peak
                    blink_end_true = trough_idx-256
                    blink_endy = trough_idx
                    blink_end = int(blink_end_true)
                end_indexes.append(blink_end)

                # plot & save each up&down phase
                # down
                plt.figure()
                plt.plot(nest)
                plt.plot(blink_starty, nest[blink_starty], 'o', color='darkorange', label='blink starts')
                plt.plot(blink_endy, nest[blink_endy], 'o', color='mediumvioletred', label='blink ends')
                plt.savefig('{}_detections_{}.png'.format(counter, n))
                plt.legend()
                plt.close()

                #calculate amplitude
                blink_amp = rev_peak_val - rev_trough_val
                blink_amplitude.append(blink_amp)

                # up phase duration (e.g. eye closing)
                up_duration = 256 - blink_starty  # samples
                up_ms = (up_duration/512)*1000  # milliseconds
                up_phase.append(up_ms)

                # down phase duration (e.g. eye opening)
                down_duration = blink_endy - 256  # samples
                down_ms = (down_duration/512)*1000  # milliseconds
                down_phase.append(down_ms)

                # total blink duration
                blink_dur_ms = ((up_duration + down_duration)/512)*1000  # milliseconds
                blink_dur.append(blink_dur_ms)

                # half base duration in seconds (best for reporting)
                prominence_data = (np.asarray([blink_amp]), np.asarray([blink_starty]), np.asarray([blink_endy]))
                width = scipy.signal.peak_widths(nest, peaks=[256], rel_height=0.5, prominence_data=prominence_data)
                half_base = width[0][0]
                half_base_dur.append(half_base/512)

                # calculate gradient
                # up-phase
                x1 = blink_starty
                y1 = nest[blink_starty]
                x2 = peak_idx
                y2 = nest[peak_idx]
                abs_rise_by_sample, abs_rise_by_sf = slope(x1, y1, x2, y2)
                abs_rise_by_sf_pos = abs(abs_rise_by_sf)
                up_grad.append(abs_rise_by_sample)

                # down-phase
                x1 = blink_endy
                y1 = nest[blink_endy]
                x2 = peak_idx
                y2 = nest[peak_idx]
                abs_rise_by_sample, abs_rise_by_sf = slope(x1, y1, x2, y2)
                abs_rise_by_sf_pos = abs(abs_rise_by_sf)
                down_grad.append(abs_rise_by_sample)

                if counter == (len(m))-1:
                    start_list.append([start_indexes])
            except:
                print('issue with', counter)
                up_phase.append('NaN')
                down_phase.append('NaN')
                blink_dur.append('NaN')


        # convert variables to array for mean calculations
        up_arr = np.array(up_phase).astype(np.float)
        down_arr = np.array(down_phase).astype(np.float)
        bl_dur_arr = np.array(blink_dur).astype(np.float)
        bl_amp_arr = np.array(blink_amplitude).astype(np.float)
        up_grad_arr = np.array(up_grad).astype(np.float)
        down_grad_arr = np.array(down_grad).astype(np.float)

        mean_up = np.nanmean(up_arr)
        mean_down = np.nanmean(down_arr)
        mean_dur = np.nanmean(bl_dur_arr)
        mean_amp = np.nanmean(bl_amp_arr)
        mean_up_grad = np.nanmean(up_grad_arr)
        mean_down_grad = np.nanmean(down_grad_arr)

        # create dict struct to store output values in
        inner_dict = {'pptID': pptid, 'Night': night, 'Group': group, 'Channel': n,
                      'Blink Number': total_blink, 'Blink Rate': blink_rate, 'Blink Density (s)': blink_density,
                      'Blink Amplitude (uv)': blink_amplitude, 'Blink Duration (ms)': blink_dur,
                      'UpPhase Duration (ms)': up_phase, 'DownPhase Duration (ms)': down_phase,
                      'UpPhase Gradient': up_grad_arr, 'DownPhase Gradient': down_grad_arr,
                      'Mean Blink Amplitude (uv)': mean_amp, 'Mean Blink Duration (ms)': mean_dur,
                      'Mean Up Phase Duration (ms)': mean_up, 'Mean Down Phase Duration (ms)': mean_down,
                      'Mean Up Phase Gradient': mean_up_grad, 'Mean Down Phase Gradient': mean_down_grad}

        if n not in outer_dict:
            outer_dict[n] = inner_dict

    # calculate eye blink order & time delay
    # calculate difference in peak times
    peak_diff_samp = []
    zip_peaks = zip(start_list[0][0], start_list[1][0])
    for list1_i, list2_i in zip_peaks:
        peak_diff_samp.append(list1_i - list2_i)
    # create var for delay in samples
    right_first_delay = []
    left_first_delay = []
    for i in peak_diff_samp:
        if i > 0:
            msec = (i / 512) * 1000
            right_first_delay.append(msec)
        elif i < 0:
            msec = (i / 512) * 1000
            msec_abs = abs(msec)
            left_first_delay.append(msec_abs)
    # blink order: right first = 1; left first = 2; together = 3
    blink_order = []
    for i in peak_diff_samp:
        if i > 0:
            blink_order.append(1)
        elif i < 0:
            blink_order.append(2)
        elif i == 0:
            blink_order.append(3)

    right_first_arr = np.array(right_first_delay).astype(np.float)
    left_first_arr = np.array(left_first_delay).astype(np.float)
    mean_right_first = np.nanmean(right_first_arr)
    mean_left_first = np.nanmean(left_first_arr)
    # convert dictionary to dataframe and store as csv
    df = pandas.DataFrame(outer_dict)
    df_t = df.T
    df_t['Blink Order'] = [blink_order for _ in range(len(df_t))]
    df_t['Mean Right First Delay (ms)'] = mean_right_first
    df_t['Mean Left First Delay (ms)'] = mean_left_first
    df_t.to_csv('eyeblink_stats.csv', index=False)

    # for plotting to check!
    # get true index start & end times
    start_times = []
    start_zip = zip(fp1_peaks, start_indexes)
    for list1_i, list2_i in start_zip:
        start_times.append(list1_i - list2_i)

    end_times = []
    end_zip = zip(fp1_peaks, end_indexes)
    for list1_i, list2_i in end_zip:
        end_times.append(list1_i + list2_i)

    fig, ax = plt.subplots()
    ax.plot(fp1_uv, linewidth=0.8, color='dimgray', label='Fp1')
    #ax.plot(fp2_uv, linewidth=0.8, color='slateblue', label='Fp2')
    #ax.plot(e1_uv, linewidth=0.8, color='darkorange', label='E1')
    #ax.plot(e2_uv, linewidth=0.8, color='red', label='E2')
    ax.plot(fp1_peaks, fp1_uv[fp1_peaks], 'o', color='darkred', label='Fp1 Peaks')
    ax.plot(start_times, fp1_uv[start_times], 'o', color='darkorange', label='blink starts')
    ax.plot(end_times, fp1_uv[end_times], 'o', color='mediumvioletred', label='blink ends')
    #ax.plot(fp2_peaks, fp2_uv[fp2_peaks], 'x', color='darkred', label='Fp2 Peaks')
    #ax.plot(e1_peaks, e1_uv[e1_peaks], 'o', color='blue', label='E1 Peaks')
    #ax.plot(e2_peaks, e1_uv[e2_peaks], 'x', color='blue', label='E2 Peaks')
    plt.legend()

    return df_t
