# MIDI.py

import numpy as np
import matplotlib.pyplot as plt
from mido import Message, MetaMessage, MidiFile, MidiTrack
from os import walk
import os.path

def getMIDIfiles(path, end_mask = "4-4.mid"):
    '''
    Generate a list of all MIDI files with the given suffix in the sepcified folder (including subfolders).

    Parameters
    ----------
    path : str
        The path to the folder containing the files.
    end_mask : str, optional
        The suffix which all the valid files share.

    Returns
    -------
    list of str
        A list of paths to the found files.
    '''
    path_list = []
    for dirpath, dirnames, filenames in walk(path):
        for filename in [f for f in filenames if f.endswith(end_mask)]:
            path_list.append(os.path.join(dirpath, filename))
    return path_list

def analyse_filepack(folderpath, plot = False):
    '''
    Provide statistics for all music files in the specified folder.
    Generate a set of the pitches used in the dataset and calculate  the lenght of each piece.
    Print the average velocity of all notes.
    Generate a mapping of the used pitches to a dense set of integers in the form of a dictionary.
    So far only functionality for type 0 MIDI files, e.g. files containing one track is implemented.

    Parameters
    ----------
    folderpath : str
        The path to the folder containing the MIDI files.
        Subfolders are also included.
    plot : bool, optional
        If true, generate histograms for the used pitches, velocities and track times and show them in a diagramm.

    Returns
    -------
    dict
        The note dictionary containing the pitches as keys and a unique integer for each pitch in the range [0, number of pitches) as the values.
    '''
    files = getMIDIfiles(folderpath)

    tracktimes = []
    note_msg_list = []

    for j, f in enumerate(files):
        print(f'Analysing {j+1}/{len(files)} files', end = '\r')
        mid_file = MidiFile(f)

        if mid_file.type == 0:
            # Ensure only one track is present
            assert len(mid_file.tracks) == 1
            time = 0
            for msg in mid_file.tracks[0]:
                if 'note' in msg.type:
                    note_msg_list.append([msg.note, msg.velocity])
                else:
                    pass
                time += msg.time
            tracktimes.append(time)
        else:
            print(mid_file.tracks)
            raise NotImplementedError()
    print(f"Analysing done!{' '*50}")

    note_msg_arr = np.array(note_msg_list) # array with [pitches, velocities] as columns
    print(f'Average Velocity: {np.mean(note_msg_arr[:,1])}')

    # generate the note dictionary
    noteset = set(*sorted([note_msg_arr[:,0]])) # generate a set of the used pitches
    notedict = {}
    for i,n in enumerate(noteset):
        notedict[n] = i

    # generate the histogramms and plot them
    if plot:
        fig, axes = plt.subplots(1,3, figsize = (15,5))
        axes[0].hist(note_msg_arr[:,0], max(note_msg_arr[:,0]))
        axes[0].set_xlabel('Pitches')
        axes[1].hist(note_msg_arr[:,1])
        axes[1].set_xlabel('Velocities')
        axes[2].hist(tracktimes)
        axes[2].set_xlabel('Track times')

    return notedict#, note_msg_arr, tracktimes

def parseFile(filepath, notedict, bar_resolution, record_note_off = True, record_vel = True):
    '''
    Translate the MIDI file into an array representation.
    This is done by temporal quntisation towards the nearest time step.
    Please make sure that the provided note dictionary contains a key for each pitch used in the file, and that the interger representation is within the range [0, len(notedict)).
    Otherwise a KeyError resp. IndexError is raised.

    Parameters
    ----------
    filepath : str
        The path to the MIDI file.
    notedict : dict
        The note dictionary providing a integer representation for each used pitch.
    bar_resolution : int
        The number of timesteps in each bar.
    record_note_off : bool, optional
        Whether to record thenote off messages into the array.
        If true, the note_off events are written as negative values if no note_on event is present in the array at the same position.
    record_vel : bool, optional
        Whether to record the velocity information into the array.
        The velocity values are normalized into the range [0,1].
        If true, the velocity is written into the array.
        Otherwise, only a 1 is recorded for each event.

    Returns
    -------
    array_like
        An array of shape (number of bars, bar_resolution, len(notedict)) which contains the represantation of the song in the MIDI file.
    '''
    mid_file = MidiFile(filepath)
    barlength = mid_file.ticks_per_beat * 4 # the length of the bar in MIDI ticks, so far this works only for 4-4 time signatures

    #length_ticks = tracktimes[j]
    #length_bars = int(np.ceil(length_ticks / barlength))
    #shape = (len(noteset), int(np.ceil(length_bars) * elements_per_bar))

    # helper function to generate an empty bar array
    makeBarMatrix = lambda : np.zeros((bar_resolution, len(notedict)))
    bar_list = []
    bar_matrix = makeBarMatrix()

    if len(mid_file.tracks) == 1:
        track = mid_file.tracks[0]
        time = 0 # time of the current message within the track [ticks]
        bar_time = 0 # time of the current bar within the track [ticks]

        for msg in track:
            time += msg.time
            bar_time += msg.time
            if 'note' in msg.type:
                while np.round(bar_time / barlength * bar_resolution) >= bar_resolution:
                    # note is not within the current bar array, add the array to the list and create a new one
                    bar_list.append(bar_matrix)
                    bar_matrix = makeBarMatrix()
                    bar_time -= barlength

                n_ind = notedict[msg.note]
                t_ind = int(np.round(bar_time / barlength * bar_resolution))

                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        val = 1
                    else:
                        if record_note_off: val = -1
                elif msg.type == 'note_off' and record_note_off:
                    val = -1
                else:
                    val = 0

                if record_vel:
                    val = val * msg.velocity / 127 # normalisation, MIDI velocities are in the range [1, 127]

                if bar_matrix[t_ind, n_ind] <= 0: # only record the event if no note_on event is already recorded in this element
                    bar_matrix[t_ind, n_ind] = val


        if bar_matrix.any(): bar_list.append(bar_matrix) # only append the last bar array if it is not empty
        midi_array = np.array(bar_list)
    else:
        print(mid_file.tracks)
        raise NotImplementedError()

    return midi_array

def parseFolder(folderpath, notedict, bar_resolution,
                savepath = None, name = 'MIDI dataset', generate_bar_stack = False, **kwargs):
    '''
    Generate array representations of all MIDI files within a folder (including subfolders).

    Parameters
    ----------
    folderpath : str
        The path to the folder.
    notedict : dict
        The note dictionary providing a integer representation for each used pitch.
    bar_resolution : int
        The number of timesteps in each bar.
    savepath : str, optional
        If provided, save the generated arrays and the note dictionary into a .npz archive under the goven folder.
    name : str, optional
        The filename prefix for the archives.
    generate_bar_stack : bool, optional
        If true, concatenate all songs into a big array alsong the first axis, which is also saved if a savepath is provided.
        Useful for generating training sets which only depend on single bars.
    **kwargs
        Further keyword arguments which are passed to the parseFile function.

    Returns
    -------
    dict
        A dictionarry with strings identifieng the MIDI file as keys and the array representation as values.
    array_like <if generate_bar_stack == True>
        The concatenation of the array representation of all songs allong the first axis.
    '''
    files = getMIDIfiles(folderpath)

    array_dict = {}

    for i,f in enumerate(files):
        print(f'{i+1}/{len(files)} files. Processing: {f}' + ' '*25, end = '\r')
        arr = parseFile(f, notedict, bar_resolution, **kwargs)
        arrname = '_'.join(f.split('/')[1:]) # generate an identifying string for the MIDI file
        array_dict[arrname] = arr
    print('Parsing done!')

    if generate_bar_stack: dataset_stacked_bars = np.concatenate(list(array_dict.values()))

    if savepath:
        name = f'{name}_{bar_resolution}bpb'
        print(f'Saving data  @ {savepath}', end='\r')
        np.savez(f'{savepath}/{name}_tracks', notedict = notedict, **array_dict)
        if generate_bar_stack:
            np.savez(f'{savepath}/{name}_bars', notedict = notedict, bars = dataset_stacked_bars)
        print('Saving done!  ')

    if generate_bar_stack: return array_dict, dataset_stacked_bars
    return array_dict

def printFile(filepath):
    '''
    Utility function to print all messages in a given file as well as some metadata.

    Parameters
    ----------
    filepath : str
        The path to the file to be printed.
    '''
    mid_file = MidiFile(filepath)
    print(f'Filepath          : {filepath}')
    print(f'Ticks per beat [ticks]: {mid_file.ticks_per_beat}')

    time = 0
    n_msg = 0
    if len(mid_file.tracks) == 1:
        for msg in mid_file.tracks[0]:
            print(msg)
            time += msg.time
            n_msg += 1
    else:
        raise NotImplementedError()

    print(f'Total runtime [ticks]: {time}, {n_msg} messages')

def writeBarArray(bar_arr, filename, note_dict, write_note_off = True, write_vel = True):
    '''
    Write an array representation of a song into a MIDI file.
    The array has to contain values in the range [0,1], other values are clipped to that range.
    So far, a fixed header setting the name, time signature etc. is used, although this should be exposed in the future.

    Parameters
    ----------
    bar_arr : array_like
        The array representation of a song with shape (number of bars, time steps in bar, number of instruments).
    filename : str
        The filename (including folders) at which location the generated file is to be saved.
    notedict : dict
        The note dictionary providing a integer representation for each used pitch.
    write_note_off : bool, optional
        If true, write negative array elements as note_off events.
    write_vel : bool, optional
        If true, translate the elements of the array into velocity information.
        If false, write messages with a fixed velocity as defined in the avg_velocity variable in the function declaration.
    '''

    # define header messages and other constants
    # (this is currently adjusted for the google magenta groove dataset and has to be changed for other filesets)
    header = [
        MetaMessage('track_name', name = 'generated MIDI track'),
        MetaMessage('instrument_name', name='Brooklyn'),
        MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8),
        MetaMessage('key_signature', key = 'C'),
        MetaMessage('set_tempo', tempo = 588235),
        Message('control_change', channel = 9, control = 4, value = 90)
    ]
    avg_velocity = 100
    channel = 9
    bar_length = 480 * 4

    bar_resolution = bar_arr.shape[1]
    chord_length = bar_length / bar_resolution # length of a bar_array timestep [ticks]
    # dictionary that translates indices into pitches (inverse of note_dict)
    inv_note_dict = {note_dict[pitch] : pitch for pitch in note_dict.keys()}

    mid_file = MidiFile()
    mid_file.type = 0

    time = 0
    notes = []
    printed_notes = 0
    velocity = avg_velocity

    bar_arr = np.clip(bar_arr, 0, 1)

    for bar in bar_arr:
        for chord in bar:
            for i, note in enumerate(chord):
                if note > 0:
                    if write_vel: velocity = int(note * 127)
                    notes.append(Message('note_on', note = inv_note_dict[i], channel = channel, velocity = velocity, time = int(time)))
                    time = 0
                    printed_notes +=1
                elif note < 0 and write_note_off:
                    if write_vel: velocity = int(-1 * note * 127)
                    notes.append(Message('note_off', note = inv_note_dict[i], channel = channel, velocity = velocity, time = int(time)))
                    time = 0
                    printed_notes += 1
                    pass
            time += chord_length

    track = header + notes
    mid_file.tracks.append(track)

    if not filename.endswith('.mid'):
        filename += f'_{bar_resolution}bpb.mid'
    mid_file.save(filename)
