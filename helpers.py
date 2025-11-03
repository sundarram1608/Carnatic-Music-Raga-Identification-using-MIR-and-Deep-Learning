##Import Libraries
#For Dataframe & Excel handling
import openpyxl
import pandas as pd

#For Array manipulations & Audio Processing
import numpy as np
import math

#For Audio Processing
import librosa
import soundfile as sf

#For File Handling
import os
from pathlib import Path



#####################********************For Sruthi Extraction & Standardization******************#####################


#Function to get all audio file paths from the dataset directory
def get_audio_file_paths(dataset_dir='dataset'):
    """
    Walk through the dataset directory and return a list of all audio file paths.
    Hidden files (starting with '.') are ignored.
    """
    audio_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.startswith('.'):
                continue
            rel_path = os.path.join(root, file)
            audio_paths.append(rel_path)

    return audio_paths

#Function to get the original sruthi frequency of the audio files
def get_sruthi_info(audio_path):
    """
    Load an audio file and identify its default sampling rate.
    Returns the audio time series and the sampling rate.
    """
    original_sruthi_frequency_hz = []
    original_sruthi_frequency_midi = []
    original_sruthi_note = []
    audio_arrays = []
    sampling_rates = [] 
    
    for audio in audio_path:
        
        # Load the audio file with the original sampling rate
        y, sr = librosa.load(audio, sr=None)
        audio_arrays.append(y)
        sampling_rates.append(sr)   
        
        # Identifying pitches and their magnitudes        
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        
        # Get the index of the maximum magnitude
        index = magnitudes.argmax()

        # Convert the flattened index to 2D indices (row, col)
        row, col = np.unravel_index(index, magnitudes.shape)

        # Get the fundamental frequency (F0) from the corresponding pitch
        sruthi_hz = pitches[row, col]
        original_sruthi_frequency_hz.append(sruthi_hz)
        
        # Convert Hz to MIDI (for easier pitch comparison)
        sruthi_midi = librosa.hz_to_midi(sruthi_hz)
        original_sruthi_frequency_midi.append(sruthi_midi)
        
        # Convert MIDI to note name
        sruthi_note = librosa.midi_to_note(sruthi_midi)
        original_sruthi_note.append(sruthi_note)
        
    return audio_arrays, sampling_rates, original_sruthi_frequency_hz, original_sruthi_frequency_midi, original_sruthi_note


#Function to standardize the sruthi frequency
def standardize_sruthi(audio_arrays, sampling_rates, original_sruthi_frequency_hz,desired_sruthi_freq_hz = 261.63):
    """
    Standardize the sruthi frequency of audio files to a desired frequency.
    Returns a list of standardized audio time series.
    """
    audios_standardized_sruthi = []
    
    for audio, sampling_rate, original_sruthi_frequency in zip(audio_arrays, sampling_rates, original_sruthi_frequency_hz):
        
        original_sruthi_frequency_midi = librosa.hz_to_midi(original_sruthi_frequency)
        desired_sruthi_freq_midi = librosa.hz_to_midi(desired_sruthi_freq_hz)
        desired_sruthi_freq_note = librosa.midi_to_note(desired_sruthi_freq_midi)
        # Calculate the pitch shift in semitones
        # Convert both frequencies to MIDI notes and find the difference in semitones
        pitch_shift_steps = desired_sruthi_freq_midi - original_sruthi_frequency_midi
        
        # Apply pitch shift to the audio (shifting F0 to the desired tonic C4)
        audio_standardized_sruthi = librosa.effects.pitch_shift(audio, sr=sampling_rate, n_steps=pitch_shift_steps)

        audios_standardized_sruthi.append(audio_standardized_sruthi)

    return audios_standardized_sruthi, desired_sruthi_freq_hz, desired_sruthi_freq_midi, desired_sruthi_freq_note


#Function to save standardized audio files
def save_standardized_audios(audio_paths, audios_standardized_sruthi, sampling_rates, out_root="sruthi_standardized_audios", in_root="dataset", suffix="_standard_sruthi"):
    """
    Writes standardized audio arrays to WAV files, preserving the directory hierarchy
    of `dataset/...` under a new root `sruthi_standardized_audios/...`.

    Parameters
    ----------
    audio_paths : list[str|Path]
        Original audio file paths (same order/length as audios_standardized_sruthi).
    audios_standardized_sruthi : list[np.ndarray]
        Audio arrays (y) you want to write.
    sampling_rates : original sampling rates
    out_root : str
        New root folder under which to mirror the hierarchy.
    in_root : str
        Existing root name to replace (typically 'dataset').
    suffix : str
        Suffix to append before the file extension (e.g., '_standard_sruthi').
    
    Returns
    -------
    written_paths : list[str]
        List of written WAV paths (relative to current working directory).
    """
    # input checks
    if len(audio_paths) != len(audios_standardized_sruthi):
        raise ValueError("audio_paths and audios_standardized_sruthi must have the same length.")

    if sampling_rates is not None and not isinstance(sampling_rates, (int, list, tuple)):
        raise TypeError("sampling_rates must be None, an int, or a list/tuple of ints.")

    if isinstance(sampling_rates, (list, tuple)) and len(sampling_rates) != len(audio_paths):
        raise ValueError("If sampling_rates is a list/tuple, its length must match audio_paths.")

    out_root = Path(out_root)
    in_root = Path(in_root)

    written_paths = []

    for i, (orig_path, y) in enumerate(zip(audio_paths, audios_standardized_sruthi)):
        orig_path = Path(orig_path)

        # Skip hidden/system files like .DS_Store, etc.
        if orig_path.name.startswith("."):
            continue

        # Compute relative path under `in_root` (e.g., 'dataset/raaga/song.mp3' -> 'raaga/song.mp3')
        try:
            rel_under_inroot = orig_path.relative_to(in_root)
        except ValueError:
            # If the file isn't under `dataset/`, still place it under out_root keeping its parent dirs
            rel_under_inroot = orig_path

        # Build new directory under out_root mirroring the hierarchy
        out_dir = out_root / rel_under_inroot.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # New filename: original stem + suffix + .wav
        # e.g., 'Sami_Ni.mp3' -> 'Sami_Ni_standard_sruthi.wav'
        out_stem = orig_path.stem + suffix
        out_path = out_dir / f"{out_stem}.wav"

        # Determine sampling rate for this file
        if sampling_rates is None:
            # Read SR from original file header (fast; does not load full audio)
            sr = librosa.get_samplerate(str(orig_path))
        elif isinstance(sampling_rates, (list, tuple)):
            sr = int(sampling_rates[i])
        else:
            sr = int(sampling_rates)

        # Write WAV
        sf.write(str(out_path), y, sr)
        written_paths.append(str(out_path))

    return written_paths


#Function to create base dataframe with audio information
def create_base_dataframe(audio_path, 
                          original_sruthi_note, 
                          original_sruthi_frequency_hz, 
                          original_sruthi_frequency_midi, 
                          standardized_sruthi_note, 
                          standardized_sruthi_frequency_hz, 
                          standardized_sruthi_frequency_midi, 
                          sampling_rates):
    """
    Create a pandas DataFrame containing audio information,
    including raga and song name extracted from the audio_path.
    """
    # Extract raga and song name from each path
    raga_list = []
    song_name_list = []

    for path in audio_path:
        # Normalize path separators for cross-platform safety
        parts = os.path.normpath(path).split(os.sep)
        
        # Expected structure: dataset/<raga>/<song_name>.<ext>
        raga = parts[1] if len(parts) > 1 else None
        song_name = os.path.splitext(parts[-1])[0]  # remove extension
        
        raga_list.append(raga)
        song_name_list.append(song_name)
    
    # Create dataframe
    base_dataframe_std_sruthi = pd.DataFrame({
                                    "audio_path": audio_path,
                                    "raga": raga_list,
                                    "song_name": song_name_list,
                                    "original_sruthi_note": original_sruthi_note,
                                    "original_sruthi_frequency_hz_f0": original_sruthi_frequency_hz,
                                    "original_sruthi_frequency_midi": original_sruthi_frequency_midi,
                                    "standardized_sruthi_note": standardized_sruthi_note,
                                    "standardized_sruthi_frequency_hz": standardized_sruthi_frequency_hz,
                                    "standardized_sruthi_frequency_midi": standardized_sruthi_frequency_midi,
                                    "sampling_rate": sampling_rates
                                })
    base_dataframe_std_sruthi.to_excel("temp/base_dataframe_std_sruthi.xlsx", index=False)

    return base_dataframe_std_sruthi

#Function to split audios into 30 second clips
def split_audio_into_clips(audio_paths, audios_standardized_sruthi, sampling_rates, clip_duration_sec=30):
    """
    Split each audio array into multiple clips of specified duration (in seconds).
    Returns:
      all_clips: List[List[np.ndarray]]  # clips per audio
      all_audio_clip_paths: List[List[str]]  # matching path (or placeholder) per clip
      df: pd.DataFrame with columns "audio_path" and "audio_clip"
      
    """
    all_clips = []
    all_audio_clip_paths = []
    
    for path, audio, sr in zip(audio_paths, audios_standardized_sruthi, sampling_rates):
        clip_length = int(clip_duration_sec * sr) 
        total_length = len(audio)
        
        clips = []
        clip_paths = []
        
        for start in range(0, total_length, clip_length):
            end = min(start + clip_length, total_length)
            clip = audio[start:end]
            clips.append(clip)
            clip_paths.append(path)
        
        all_clips.append(clips)
        all_audio_clip_paths.append(clip_paths)
        
    flat_clips = [clip for clips in all_clips for clip in clips]
    flat_paths = [path for paths in all_audio_clip_paths for path in paths]
    
    # Create DataFrame
    clipped_df = pd.DataFrame({
                                "clip_number": [i + 1 for clips in all_clips for i in range(len(clips))],
                                "audio_path": flat_paths,
                                "audio_clip": flat_clips
                            })
                    
    return clipped_df

#Function to merge base dataframe details into clipped dataframe
def merge_details_to_clip(clipped_df, base_dataframe_std_sruthi):
    """
    Merge the base dataframe details into the clipped dataframe based on audio_path.
    """
    clipped_df_std_sruthi_merged = pd.merge(clipped_df, base_dataframe_std_sruthi, on="audio_path", how="left")
    clipped_df_std_sruthi_merged.to_excel("temp/clipped_df_std_sruthi_merged.xlsx", index=False)
    
    return clipped_df_std_sruthi_merged

#Function to run the entire sruthi standardization pipeline
def run_sruthi_standardization_pipeline():
    # Step 1: Get audio file paths
    audio_path = get_audio_file_paths()    
    # Step 2: Get original sruthi information
    audio_arrays, sampling_rates, original_sruthi_frequency_hz, original_sruthi_frequency_midi, original_sruthi_note = get_sruthi_info(audio_path)
    # Step 3: Standardize sruthi frequency
    audios_standardized_sruthi, std_sruthi_freq_hz, std_sruthi_freq_midi, std_sruthi_note = standardize_sruthi(audio_arrays, sampling_rates, original_sruthi_frequency_hz)
    # Step 4: Save standardized audio files
    # standardized_audio_paths = save_standardized_audios(audio_paths, audios_standardized_sruthi, sampling_rates, out_root=out_root, in_root=dataset_dir)
    # Step 5: Create base dataframe
    base_dataframe_std_sruthi = create_base_dataframe(audio_path, original_sruthi_note, original_sruthi_frequency_hz, original_sruthi_frequency_midi, std_sruthi_note, std_sruthi_freq_hz, std_sruthi_freq_midi, sampling_rates)
    # Step 6: Split audios into clips
    clipped_df = split_audio_into_clips(audio_path, audios_standardized_sruthi, sampling_rates, clip_duration_sec=30)
    # Step 7: Merge details to clipped dataframe
    clipped_df_std_sruthi_merged = merge_details_to_clip(clipped_df, base_dataframe_std_sruthi)
    
    return clipped_df_std_sruthi_merged