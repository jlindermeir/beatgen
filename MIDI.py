import numpy as np
import matplotlib.pyplot as plt
from mido import Message, MetaMessage, MidiFile, MidiTrack
from os import walk
import os.path

def getMIDIfiles(path, end_mask = "4-4.mid"):
    path_list = []
    for dirpath, dirnames, filenames in walk(path):
        for filename in [f for f in filenames if f.endswith(end_mask)]:
            path_list.append(os.path.join(dirpath, filename))
    return path_list

def analyse_filepack(folderpath, plot = False):
    files = getMIDIfiles(folderpath)
    # provide statisitcs for all music files, generate a set of the notes uses in the dataset and calculate lenght of each piece

    #notes = []
    times = []
    tracktimes = []

    note_msg_list = []

    for j, f in enumerate(files):
        print(f'Analysing {j+1}/{len(files)} files', end = '\r')
        mid_file = MidiFile(f)

        if mid_file.type == 0:
            # Only one track in file:
            assert len(mid_file.tracks) == 1
            time = 0
            for msg in mid_file.tracks[0]:
                if 'note' in msg.type:
                    note_msg_list.append([msg.note, msg.velocity])
                    #notes.append(msg.note)
                else:
                    pass
                #print(msg)
                time += msg.time
            tracktimes.append(time)
    print(f"Analysing done!{' '*50}")

    note_msg_arr = np.array(note_msg_list)
    print(f'Averagae Velocity: {np.mean(note_msg_arr[:,1])}')
    noteset = set(*sorted([note_msg_arr[:,0]]))
    notedict = {}
    for i,n in enumerate(noteset):
        notedict[n] = i

    if plot:
        fig, axes = plt.subplots(1,3, figsize = (30,10))
        axes[0].hist(note_msg_arr[:,0], max(note_msg_arr[:,0]))
        axes[0].set_xlabel('Pitches')
        axes[1].hist(note_msg_arr[:,1])
        axes[1].set_xlabel('Velocities')
        axes[2].hist(tracktimes)
        axes[2].set_xlabel('Track times')


    return notedict, note_msg_arr, tracktimes

def parseFile(filepath, notedict, bar_resolution, record_note_off = False):
    mid_file = MidiFile(filepath)
    barlength = mid_file.ticks_per_beat * 4
    velocity_cutoff = 0

    #length_ticks = tracktimes[j]
    #length_bars = int(np.ceil(length_ticks / barlength))
    #shape = (len(noteset), int(np.ceil(length_bars) * elements_per_bar))

    makeBarMatrix = lambda : np.zeros((bar_resolution, len(notedict)))
    bar_list = []
    bar_matrix = makeBarMatrix()

    if len(mid_file.tracks) == 1:
        track = mid_file.tracks[0]
        time = 0 # time of the current message within the track [ticks]
        bar_time = 0 # time of the current bar within the track [ticks]

        for msg in track:
            if 'note' in msg.type:
                if msg.velocity < velocity_cutoff:
                    continue
                while bar_time >= barlength:
                    bar_list.append(bar_matrix)
                    bar_matrix = makeBarMatrix()
                    bar_time -= barlength

                n_ind = notedict[msg.note]
                t_ind = int(np.floor(bar_time / barlength * bar_resolution))

                #val = 1 if ('on' in msg.type) else -1
                if msg.type == 'note_on':
                    val = 1
                elif msg.type == 'note_off' and record_note_off:
                    val = -1
                else: 
                    val = 0
                if bar_matrix[t_ind, n_ind] <= 0:
                    bar_matrix[t_ind, n_ind] = val
            time += msg.time
            bar_time += msg.time
        
        if bar_matrix.any(): bar_list.append(bar_matrix)
        midi_array = np.array(bar_list)
    else:
        print(mid_file.tracks)
        raise NotImplementedError()

    return midi_array

def parseFolder(folderpath, notedict, bar_resolution, 
                savepath = None, name = 'MIDI dataset', record_note_off = False, generate_bar_stack = False):
    files = getMIDIfiles(folderpath)

    array_dict = {}

    for i,f in enumerate(files):
        print(f'{i+1}/{len(files)} files. Processing: {f}' + ' '*25, end = '\r')
        arr = parseFile(f, notedict, bar_resolution, record_note_off = record_note_off)
        arrname = '_'.join(f.split('/')[1:])
        array_dict[arrname] = arr
    print('Parsing done!')

    if generate_bar_stack: dataset_stacked_bars = np.concatenate(list(array_dict.values()))

    if savepath:
        print(f'Saving data  @ {savepath}', end='\r')
        np.savez(f'{savepath}/{name}_tracks', notedict = notedict, **array_dict)
        if generate_bar_stack:
            np.savez(f'{savepath}/{name}_bars', notedict = notedict, bars = dataset_stacked_bars)
        print('Saving done!  ')

    if generate_bar_stack: return array_dict, dataset_stacked_bars
    return array_dict

def printFile(filepath):
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
    
def writeBarArray(bar_arr, filename, note_dict):
    # bar_arr : shape (n_bars, bar_res, n_instr)
    print(bar_arr.shape)
    header = [
        MetaMessage('track_name', name = 'genereated MIDI track'),
        MetaMessage('instrument_name', name='Brooklyn'),
        MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8),
        MetaMessage('key_signature', key = 'C'),
        MetaMessage('set_tempo', tempo = 588235),
        Message('control_change', channel = 9, control = 4, value = 90)
    ]
    velocity = 63
    channel = 9
    bar_length = 480 * 4
    
    bar_resolution = bar_arr.shape[1]
    
    chord_length = bar_length / bar_resolution
    inv_note_dict = {note_dict[pitch] : pitch for pitch in note_dict.keys()}
    
    mid_file = MidiFile()
    mid_file.type = 0
    
    time = 0
    notes = []
    printed_notes = 0
    
    for bar in bar_arr:
        for chord in bar:
            for i, note in enumerate(chord):
                if note == 1:
                    notes.append(Message('note_on', note = inv_note_dict[i], channel = channel, velocity = velocity, time = int(time)))
                    time = 0
                    printed_notes +=1
                elif note == -1:
                    notes.append(Message('note_off', note = inv_note_dict[i], channel = channel, velocity = velocity, time = int(time)))
                    time = 0
                    printed_notes += 1
                    pass
            time += chord_length
            
    
    track = header + notes
    print(printed_notes)
    
    mid_file.tracks.append(track)
    
    if not filename.endswith('.mid'):
        filename += f'_{bar_resolution}bpb.mid'
    
    mid_file.save(filename)