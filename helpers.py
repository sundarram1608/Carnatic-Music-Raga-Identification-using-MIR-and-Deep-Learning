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
from scipy.signal import find_peaks

#For File Handling
import os
from pathlib import Path


#####################********************For Sruthi Extraction & Standardization******************#####################

#-------------------------------------------------------------------------
#Function to get all audio file paths from the dataset directory
#-------------------------------------------------------------------------

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

#-------------------------------------------------------------------------
#Function to get the original sruthi frequency of the audio files using argmax of magnitudes
# This can be used as a simpler alternative to the pitchclass-based method but is not right always and relies more on chance
#-------------------------------------------------------------------------
def get_sruthi_info_piptrack_argmax(audio_path):
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
        
        # Load the audio file with the original sampling rate (y is a NumPy array of floats. It contains the raw amplitude values of the audio signal at each time step.)
        y, sr = librosa.load(audio, sr=None)
        audio_arrays.append(y)
        sampling_rates.append(sr)   
        
        # Identifying pitches and their magnitudes (Uses HPS - Harmonic Product Spectrum method)        
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

#-------------------------------------------------------------------------
#Function to get the original sruthi frequency of the audio files using argmax of magnitudes
#-------------------------------------------------------------------------
def get_sruthi_info_piptrack_pitchclass(audio_path):
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
        
        # Load the audio file with the original sampling rate (y is a NumPy array of floats. It contains the raw amplitude values of the audio signal at each time step.)
        y, sr = librosa.load(audio, sr=None)
        audio_arrays.append(y)
        sampling_rates.append(sr)   
        
        # Identifying pitches and their magnitudes (Uses HPS - Harmonic Product Spectrum method)        
        pitch_values, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        
        # Extract pitch per time frame using max magnitude
        pitch_track = []

        for t in range(pitch_values.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitch_values[idx, t]
            pitch_track.append(pitch)
            
        pitch_track = np.array(pitch_track)

        # Remove zero or negative values from the pitch track
        # This is important because librosa may return zero for frames where no pitch was detected            
        pitch_track = pitch_track[pitch_track > 0]

        # Convert pitch track to cents relative to a reference pitch (C3 = 130.8128 Hz)
        ref = librosa.note_to_hz("C3")   # 130.8128 Hz
        pitch_cents = 1200 * np.log2(pitch_track / ref)

        # Compute pitch class histogram
        pitch_class = np.mod(pitch_cents, 1200)
        hist, bins = np.histogram(pitch_class, bins=120)  # 10 cent resolution
        bin_centers = 0.5*(bins[:-1] + bins[1:])

        # Peak Detection
        peaks, props = find_peaks(hist, height=np.max(hist) * 0.4)
        if len(peaks) == 0:
            pass
        else:
            tonic_pc = bin_centers[peaks[0]]  # take the highest peak
            # print("Estimated Tonic Pitch Class (in cents):", tonic_pc)

        # Convert tonic pitch class back to frequency in Hz        
        tonic_hz = ref * 2 ** (tonic_pc / 1200) # This is the orignal Sruthi Frequency in Hz (F0)
        original_sruthi_frequency_hz.append(tonic_hz)
        
        # Convert Hz to MIDI (for easier pitch comparison)
        sruthi_midi = librosa.hz_to_midi(tonic_hz)
        original_sruthi_frequency_midi.append(sruthi_midi)
        
        # Convert MIDI to note name
        sruthi_note = librosa.midi_to_note(sruthi_midi)
        original_sruthi_note.append(sruthi_note)
        
    return audio_arrays, sampling_rates, original_sruthi_frequency_hz, original_sruthi_frequency_midi, original_sruthi_note

#-------------------------------------------------------------------------
#Function to standardize the sruthi frequency
#-------------------------------------------------------------------------
def standardize_sruthi(audio_arrays, sampling_rates, original_sruthi_frequency_hz,desired_sruthi_freq_hz = librosa.note_to_hz("C3")):
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
        
        # Apply pitch shift to the audio (shifting F0 to the desired tonic C3)
        audio_standardized_sruthi = librosa.effects.pitch_shift(audio, sr=sampling_rate, n_steps=pitch_shift_steps)

        audios_standardized_sruthi.append(audio_standardized_sruthi)

    return audios_standardized_sruthi, desired_sruthi_freq_hz, desired_sruthi_freq_midi, desired_sruthi_freq_note

#-------------------------------------------------------------------------
#Function to save standardized audio files
#-------------------------------------------------------------------------
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

#-------------------------------------------------------------------------
#Function to create base dataframe with audio information
#-------------------------------------------------------------------------
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
    os.makedirs("temp", exist_ok=True)
    # base_dataframe_std_sruthi.to_excel("temp/1. base_dataframe_std_sruthi.xlsx", index=False)
    base_dataframe_std_sruthi = base_dataframe_std_sruthi.reset_index(drop=True)
    # base_dataframe_std_sruthi.to_pickle("temp/1. base_dataframe_std_sruthi.pkl")

    return base_dataframe_std_sruthi

#-------------------------------------------------------------------------
#Function to split audios into 30 second clips
#-------------------------------------------------------------------------
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

#-------------------------------------------------------------------------
#Function to merge base dataframe details into clipped dataframe
#-------------------------------------------------------------------------
def merge_details_to_clip(clipped_df, base_dataframe_std_sruthi):
    """
    Merge the base dataframe details into the clipped dataframe based on audio_path.
    """
    clipped_df_std_sruthi_merged = pd.merge(clipped_df, base_dataframe_std_sruthi, on="audio_path", how="left")
    # clipped_df_std_sruthi_merged.to_excel("temp/2. clipped_df_std_sruthi_merged.xlsx", index=False)
    clipped_df_std_sruthi_merged = clipped_df_std_sruthi_merged.reset_index(drop=True)
    # clipped_df_std_sruthi_merged.to_pickle("temp/2. clipped_df_std_sruthi_merged.pkl")
    # clipped_df_std_sruthi_merged.to_parquet("temp/2. clipped_df_std_sruthi_merged.parquet")

    
    return clipped_df_std_sruthi_merged




#####################********************For Raga Feature Extraction******************#####################

#-------------------------------------------------------------------------
#Functions to Extract Pitch Contour in Hz & Cents, Pitch-class sequence (mod 1200), corresponding time stamps and store them to dataframe
#Pitch Contour in Hz & Cents captures Characteristic phrases (prayogas), Arohana-Avarohana, Contour shape, Gamakas (partially)
#Pitch-class sequence (mod 1200) captures Swaras used, Presence/absence of specific swaras, Relative prominence, Arohana/Avarohana sequence, Swaras in prayogas
#-------------------------------------------------------------------------
def extract_pitch_features_from_clip(audio_clip, sampling_rate, ref_note="C3", fmin=60.0, fmax=1000.0):
    
    """
    This function extracts pitch contour and pitch-class.
    Both are merged together to avoid recomputation of pyin.

    Extract the pitch contour (f0) from a standardized audio clip
    and express it in cents relative to a reference note (default C3).
    Pitch Contour (f₀) → The melody line, changing over time


    Parameters
    ----------
    audio_clip : np.ndarray
        1D numpy array containing the audio samples for a clip
        (already sruthi-standardized in your pipeline).
    sampling_rate : int
        Sampling rate (sr) of the audio clip.
    ref_note : str, optional
        Reference note for cents calculation (default: "C3").
        Since your pipeline standardizes Sruthi to C3, this is a natural choice.
    fmin : float, optional
        Minimum expected f0 in Hz for pyin (default: 60 Hz).
    fmax : float, optional
        Maximum expected f0 in Hz for pyin (default: 1000 Hz).

    Returns
    -------
    f0_hz : np.ndarray
        1D array of fundamental frequency estimates in Hz.
        Unvoiced frames are np.nan.
    f0_cents : np.ndarray
        1D array of f0 in cents relative to `ref_note`.
        Unvoiced frames are np.nan.
    times : np.ndarray
        1D array of time stamps (in seconds) corresponding to each f0 estimate.
    pitch_class : np.ndarray
        1D array of pitch_class in cents relative to `ref_note`.
        Unvoiced frames are np.nan.
    """

    # --- 1. Estimate f0 using librosa.pyin ---
    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
                                                    audio_clip,
                                                    fmin=fmin,
                                                    fmax=fmax,
                                                    sr=sampling_rate,
                                                )
    # f0_hz is 1D array with np.nan where pitch is unvoiced

    # --- 2. Build time axis for each frame ---
    times = librosa.times_like(f0_hz, sr=sampling_rate)


    # --- 3. Convert f0 from Hz to cents relative to ref_note ---
    ref_freq = librosa.note_to_hz(ref_note)  # C3 ≈ 130.81 Hz

    # Avoid warnings when taking log of nan
    f0_cents = np.full_like(f0_hz, fill_value=np.nan, dtype=float)

    # Mask valid (voiced) frames
    valid_mask = ~np.isnan(f0_hz)

    # Cents conversion: 1200 * log2(f / ref_freq)
    f0_cents[valid_mask] = 1200.0 * np.log2(f0_hz[valid_mask] / ref_freq)

    # --- 4. Compute pitch-class sequence (mod 1200) ---
    pitch_class = np.full_like(f0_cents, fill_value=np.nan)
    pitch_class[valid_mask] = np.mod(f0_cents[valid_mask], 1200.0)


    return f0_hz, f0_cents,pitch_class, times

# Function to create dataframe from exteracted pitch features
def add_pitch_features_to_dataframe(clipped_df_std_sruthi_merged):
    """
    Takes the clipped_df_std_sruthi_merged dataframe and
    adds f0_hz, f0_cents, and f0_times columns.

    Each row corresponds to one standardized audio clip.
    """

    pitch_features_added_df = clipped_df_std_sruthi_merged.copy()
    
    f0_hz_list = []
    f0_cents_list = []
    f0_times_list = []
    pitch_class_list = []

    for idx, row in pitch_features_added_df.iterrows():
        audio_clip = row["audio_clip"]
        sr = row["sampling_rate"]

        # Extract f0 features
        f0_hz, f0_cents, pitch_class, times = extract_pitch_features_from_clip(
                                                                                audio_clip=audio_clip,
                                                                                sampling_rate=sr
                                                                            )

        f0_hz_list.append(f0_hz)
        f0_cents_list.append(f0_cents)
        f0_times_list.append(times)
        pitch_class_list.append(pitch_class)

    # Store as new columns
    pitch_features_added_df["f0_hz"] = f0_hz_list
    pitch_features_added_df["f0_cents"] = f0_cents_list
    pitch_features_added_df["f0_times"] = f0_times_list
    pitch_features_added_df["pitch_class"] = pitch_class_list
    
    # pitch_features_added_df.to_excel("temp/3. pitch_features_added_df.xlsx", index=False)
    pitch_features_added_df = pitch_features_added_df.reset_index(drop=True)
    # pitch_features_added_df.to_pickle("temp/3. pitch_features_added_df.pkl")

    return pitch_features_added_df

#-------------------------------------------------------------------------
# Function: Extract Interval Sequence from f0_cents
# Interval sequence captures Arohana/Avarohana dynamics, characteristic transitions, and gamaka movement patterns.
#-------------------------------------------------------------------------
def extract_interval_sequence(f0_cents):
    """
    Compute the interval (successive pitch differences) from the f0_cents contour.

    Interval[t] = f0_cents[t] - f0_cents[t-1]

    Parameters
    ----------
    f0_cents : np.ndarray
        1D array of pitch contour in cents relative to C3.
        Unvoiced frames should be np.nan.

    Returns
    -------
    intervals : np.ndarray
        1D array of successive pitch differences (in cents).
        Length = len(f0_cents) - 1.
        Intervals involving np.nan are set to np.nan.
    """

    # Shift f0 sequence by one frame to compute differences
    f0_prev = f0_cents[:-1]
    f0_next = f0_cents[1:]

    # Initialize interval array
    intervals = np.full_like(f0_prev, fill_value=np.nan, dtype=float)

    # Valid frames = both are not nan
    valid_mask = (~np.isnan(f0_prev)) & (~np.isnan(f0_next))

    # Compute differences only for valid frames
    intervals[valid_mask] = f0_next[valid_mask] - f0_prev[valid_mask]

    return intervals

# Function to create dataframe from exteracted interval sequence
def add_interval_sequence_to_dataframe(pitch_features_df):
    """
    Adds interval sequence for each clip to the dataframe.
    Requires f0_cents column to be present.
    """

    df = pitch_features_df.copy()
    interval_list = []

    for idx, row in df.iterrows():
        f0_cents = row["f0_cents"]
        intervals = extract_interval_sequence(f0_cents)
        interval_list.append(intervals)

    df["interval_sequence"] = interval_list

    # df.to_excel("temp/4. interval_features_added_df.xlsx", index=False)
    df = df.reset_index(drop=True)
    # df.to_pickle("temp/4. interval_features_added_df.pkl")

    return df

#-------------------------------------------------------------------------
# Function: Extract Derivative (Δf0) and Second Derivative (Δ²f0) of f0 from f0_hz
# Δf0 captures gamaka movement, vibrato speed, and instantaneous pitch slope.
# Δ²f0 captures curvature of gamakas: acceleration/deceleration in pitch movement.
#-------------------------------------------------------------------------
def extract_f0_derivative_and_second_derivative(f0_hz):
    """
    Compute the derivative (successive differences) of the f0 contour in Hz.
    Δf0[t] = f0_hz[t] - f0_hz[t-1]

    Also Computes the second derivative (curvature) of the f0 contour in Hz.
    Δ²f0[t] = (f0[t+1] - f0[t]) - (f0[t] - f0[t-1])
            = Δf0[t] - Δf0[t-1]

    Parameters
    ----------
    f0_hz : np.ndarray
        1D array of raw f0 contour in Hz.
        Contains np.nan for unvoiced frames.

    Returns
    -------
    f0_second_derivative : np.ndarray
        1D array of second derivative values (Hz/frame²).
        Length = len(f0_hz) - 2.
        Frames involving np.nan are set to np.nan.
    
    Difference Between Interval Sequence & Δf₀:
        ✔ Interval sequence
            Computed from f₀ in cents
            Represents musical intervals, swara steps, arohana-avarohana transitions
            Units: cents
        ✔ Derivative sequence (Δf₀)
            Computed from f₀ in Hz
            Represents speed of pitch movement, “gamaka motion”, “glides”
            Units: Hz per frame
            This directly captures:
                - gamaka speed
                - phrase bends
                - vibrato rates
                - microtonal slopes
    Parameters
    ----------
    f0_hz : np.ndarray
        1D array of raw pitch contour (Hz) from pyin.
        Unvoiced frames should be np.nan.

    Returns
    -------
    f0_derivative : np.ndarray
        1D array of instantaneous pitch changes (in Hz).
        Length = len(f0_hz) - 1.
        Values involving np.nan are set to np.nan.
        
    f0_second_derivative : np.ndarray
        1D array of second derivative values (Hz/frame²).
        Length = len(f0_hz) - 2.
        Frames involving np.nan are set to np.nan.

    """

    # Shift by 1 frame to compute differences
    f0_prev = f0_hz[:-1]
    f0_next = f0_hz[1:]

    # Output array
    f0_derivative = np.full_like(f0_prev, fill_value=np.nan, dtype=float)

    # Valid frames = both values are not NaN
    valid_mask = (~np.isnan(f0_prev)) & (~np.isnan(f0_next))

    # Compute Δf0
    f0_derivative[valid_mask] = f0_next[valid_mask] - f0_prev[valid_mask]
    
    # Now compute second derivative based on first derivative:
    fd_prev = f0_derivative[:-1]
    fd_next = f0_derivative[1:]
    
    f0_second_derivative = np.full_like(fd_prev, fill_value=np.nan, dtype=float)
    valid_mask_2 = (~np.isnan(fd_prev)) & (~np.isnan(fd_next))
    f0_second_derivative[valid_mask_2] = fd_next[valid_mask_2] - fd_prev[valid_mask_2]

    return f0_derivative, f0_second_derivative

# Function to create dataframe from extracted Δf0 (derivative of f0)
def add_f0_derivative_and_second_derivative_to_dataframe(interval_features_added_df):
    """
    Adds Δf0 (derivative of the f0_hz contour) and Δ²f0 (second derivative of f0_hz contour) to the dataframe.
    Requires f0_hz column to be already present in the dataframe.
    """

    df = interval_features_added_df.copy()
    f0_derivative_list = []
    f0_second_derivative_list = []

    for idx, row in df.iterrows():
        f0_hz = row["f0_hz"]
        f0_derivative,f0_second_derivative = extract_f0_derivative_and_second_derivative(f0_hz)
        f0_derivative_list.append(f0_derivative)
        f0_second_derivative_list.append(f0_second_derivative)

    df["f0_derivative"] = f0_derivative_list
    df["f0_second_derivative"] = f0_second_derivative_list

    # Save results
    # df.to_excel("temp/5. f0_derivative_and_second_derivative_added_df.xlsx", index=False)
    df = df.reset_index(drop=True)
    # df.to_pickle("temp/5. f0_derivative_and_second_derivative_added_df.pkl")

    return df

#-------------------------------------------------------------------------
# Gamaka Descriptors Extraction
# (1) extent  - How far the pitch oscillates around its center.(Extent of Gamaka) - depth of oscillation (kampita), bends (jaarus).
#               Compute → Standard deviation of f₀ (Hz or cents) over a sliding window
#                       → Max–min range in cents
# (2) rate    - how fast oscillations happen
#                 → Zero-crossing rate (ZCR) of the Δf₀ (derivative) signal
#                   (A zero crossing ≈ one oscillation cycle)
# (3) modulation index - strength of oscillation - Raga character — e.g., Bhairavi wide gamaka vs. Kalyani mild gamaka.

#-------------------------------------------------------------------------
def extract_gamaka_descriptors(f0_cents, f0_derivative, window_size=50):
    """
    Extract gamaka descriptors from the f0 contour in cents and Δf0 in Hz.

    Parameters
    ----------
    f0_cents : np.ndarray
        1D array of f0 in cents (nan for unvoiced)
    f0_derivative : np.ndarray
        1D array of Δf0 (Hz), length = len(f0_hz)-1
    window_size : int
        Number of frames for sliding analysis (≈ 50–100 ms typical)

    Returns
    -------
    gamaka_extent : np.ndarray
        Sliding-window standard deviation of f0_cents (local pitch spread)

    gamaka_rate : np.ndarray
        Zero-crossing count of Δf0 over each window (oscillation speed)

    modulation_index : np.ndarray
        gamaka_extent / (gamaka_rate + small_eps)
    """

    N = len(f0_cents)

    # Prepare outputs
    gamaka_extent = np.full(N, np.nan, dtype=float)
    gamaka_rate   = np.full(N, np.nan, dtype=float)
    modulation_index = np.full(N, np.nan, dtype=float)

    eps = 1e-6

    # Iterate over windows
    for i in range(window_size, N - window_size):
        # Window slices
        f0_win = f0_cents[i-window_size:i+window_size]
        d_win  = f0_derivative[i-window_size:i+window_size-1]

        # Remove nan values
        f0_valid = f0_win[~np.isnan(f0_win)]
        d_valid  = d_win[~np.isnan(d_win)]

        if len(f0_valid) < 5 or len(d_valid) < 5:
            continue

        # 1. EXTENT = local std dev of pitch (cents)
        gamaka_extent[i] = np.std(f0_valid)

        # 2. RATE = zero-crossings of Δf0
        zero_crossings = np.where(np.diff(np.sign(d_valid)))[0]
        gamaka_rate[i] = len(zero_crossings)

        # 3. MODULATION INDEX
        modulation_index[i] = gamaka_extent[i] / (gamaka_rate[i] + eps)

    return gamaka_extent, gamaka_rate, modulation_index

# Function to create dataframe from extracted Gamaka features
def add_gamaka_descriptors_to_dataframe(df_with_derivatives):
    """
    Adds gamaka_extent, gamaka_rate, modulation_index to the dataframe.
    Requires f0_cents and f0_derivative columns to exist.
    """

    df = df_with_derivatives.copy()
    extent_list = []
    rate_list = []
    mod_index_list = []

    for idx, row in df.iterrows():
        f0_cents = row["f0_cents"]
        f0_derivative = row["f0_derivative"]

        extent, rate, mod_idx = extract_gamaka_descriptors(
                                    f0_cents=f0_cents,
                                    f0_derivative=f0_derivative,
                                    window_size=50
                                )
        extent_list.append(extent)
        rate_list.append(rate)
        mod_index_list.append(mod_idx)

    df["gamaka_extent"] = extent_list
    df["gamaka_rate"] = rate_list
    df["modulation_index"] = mod_index_list

    # df.to_excel("temp/6. gamaka_descriptors_added_df.xlsx", index=False)
    df = df.reset_index(drop=True)
    # df.to_pickle("temp/6. gamaka_descriptors_added_df.pkl")

    return df

#-------------------------------------------------------------------------
# Mel-spectrogram Extraction (128-bin)
# Captures timbre, harmonic envelope, formants, and steady-state spectral content.
#-------------------------------------------------------------------------
def extract_melspectrogram_128bins(audio_clip,sampling_rate,n_mels=128,n_fft=2048,hop_length=512,fmin=20,fmax=None):
    """
    Compute a 128-bin Mel-spectrogram for a standardized audio clip.

    Parameters
    ----------
    audio_clip : np.ndarray
        1D audio array (sruthi-standardized in your pipeline).
    sampling_rate : int
        Sampling rate of the audio.
    n_mels : int
        Number of mel bands (default: 128).
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length for STFT.
    fmin : float
        Minimum frequency (Hz).
    fmax : float or None
        Maximum frequency. None = sr/2.

    Returns
    -------
    mel_db : np.ndarray
        2D array of Mel-spectrogram in dB units.
        Shape: (n_mels, T)
    """

    # 1. Compute Mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(
                                                y=audio_clip,
                                                sr=sampling_rate,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                n_mels=n_mels,
                                                fmin=fmin,
                                                fmax=fmax
                                            )

    # 2. Convert amplitudes to logarithmic dB scale
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_db

# Function to create dataframe from melspectrograms
def add_melspectrogram_to_dataframe(df_with_pitch_features):
    """
    Adds a 128-bin Mel-spectrogram matrix for each audio clip.
    Stores each spectrogram as a 2D numpy array (n_mels × time_frames).
    """

    df = df_with_pitch_features.copy()
    mel_list = []

    for idx, row in df.iterrows():
        audio_clip = row["audio_clip"]
        sr = row["sampling_rate"]

        mel_db = extract_melspectrogram_128bins(
                                                    audio_clip=audio_clip,
                                                    sampling_rate=sr
                                            )
        mel_list.append(mel_db)

    df["mel_spectrogram_128"] = mel_list

    # df.to_excel("temp/7. mel_spectrogram_added_df.xlsx", index=False)
    df = df.reset_index(drop=True)
    # df.to_pickle("temp/7. mel_spectrogram_added_df.pkl")

    return df

#-------------------------------------------------------------------------
# MFCC Extraction (13 coefficients + delta + delta-delta)
# Captures formants, vocal-tract shape, articulation, brightness,
# and timbre-related cues important for raga identity.
#-------------------------------------------------------------------------
def extract_mfcc_features(audio_clip,sampling_rate,n_mfcc=13,n_fft=2048,hop_length=512):
    """
    Extract MFCC, delta MFCC, and delta-delta MFCC features from the audio clip.

    Parameters
    ----------
    audio_clip : np.ndarray
        1D standardized audio clip.
    sampling_rate : int
        Sampling rate of the audio.
    n_mfcc : int
        Number of MFCC coefficients (default: 13).
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length for STFT.

    Returns
    -------
    mfcc : np.ndarray
        Shape: (n_mfcc, T)
    delta_mfcc : np.ndarray
        1st derivative of MFCC (n_mfcc, T)
    delta2_mfcc : np.ndarray
        2nd derivative of MFCC (n_mfcc, T)
    """

    # 1. Compute MFCCs
    mfcc = librosa.feature.mfcc(
                                    y=audio_clip,
                                    sr=sampling_rate,
                                    n_mfcc=n_mfcc,
                                    n_fft=n_fft,
                                    hop_length=hop_length
                                )

    # 2. Compute Delta MFCC (first derivative)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)

    # 3. Compute Delta-Delta MFCC (second derivative)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    return mfcc, delta_mfcc, delta2_mfcc

# Function to create dataframe from extracted MFCC features
def add_mfcc_features_to_dataframe(df_with_melspectrogram_features):
    """
    Adds MFCC, delta MFCC, and delta-delta MFCC features to the dataframe.
    """

    df = df_with_melspectrogram_features.copy()

    mfcc_list = []
    delta_list = []
    delta2_list = []

    for idx, row in df.iterrows():
        audio_clip = row["audio_clip"]
        sr = row["sampling_rate"]

        mfcc, d_mfcc, dd_mfcc = extract_mfcc_features(
                                                        audio_clip=audio_clip,
                                                        sampling_rate=sr
                                                    )

        mfcc_list.append(mfcc)
        delta_list.append(d_mfcc)
        delta2_list.append(dd_mfcc)

    df["mfcc"] = mfcc_list
    df["mfcc_delta"] = delta_list
    df["mfcc_delta2"] = delta2_list

    # Save outputs
    # df.to_excel("temp/8. mfcc_features_added_df.xlsx", index=False)
    df = df.reset_index(drop=True)
    # df.to_pickle("temp/8. mfcc_features_added_df.pkl")

    return df

#-------------------------------------------------------------------------
# Chroma Feature Extraction (12-bin pitch class energy)
# Captures swara usage, prominence patterns, raga scale structure, and octave-invariant melodic distribution.
# Chroma features represent pitch class energy distribution across the 12 semitone bins
#-------------------------------------------------------------------------
def extract_chroma_features(audio_clip,sampling_rate,n_fft=2048,hop_length=512):
    """
    Compute 12-bin chroma feature representation for the audio clip.

    Parameters
    ----------
    audio_clip : np.ndarray
        1D standardized audio clip.
    sampling_rate : int
        Sampling rate.
    n_fft : int
        FFT window size.
    hop_length : int
        STFT hop length.

    Returns
    -------
    chroma : np.ndarray
        12 x T chromagram matrix (each row = one pitch class).
    """

    # Compute chroma using energy normalized STFT
    chroma = librosa.feature.chroma_stft(
                                            y=audio_clip,
                                            sr=sampling_rate,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            norm=2  # L2 norm makes features comparable across clips
                                        )

    return chroma

# Function to create dataframe from extracted Chroma features
def add_chroma_features_to_dataframe(mfcc_features_added_df):
    """
    Adds 12-bin chroma features (chroma_stft) to the dataframe.
    Each row receives a 12 × T chroma matrix.
    """

    df = mfcc_features_added_df.copy()
    chroma_list = []

    for idx, row in df.iterrows():
        audio_clip = row["audio_clip"]
        sr = row["sampling_rate"]

        chroma = extract_chroma_features(
                                            audio_clip=audio_clip,
                                            sampling_rate=sr
                                        )
        chroma_list.append(chroma)

    df["chroma_12"] = chroma_list

    df.to_excel("temp/9. chroma_features_added_df.xlsx", index=False)
    df = df.reset_index(drop=True)
    df.to_pickle("temp/9. chroma_features_added_df.pkl")

    return df
