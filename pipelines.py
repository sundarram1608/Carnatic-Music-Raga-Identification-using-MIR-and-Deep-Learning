from helpers import *

#-------------------------------------------------------------------------
#Function to run the entire sruthi standardization pipeline
#-------------------------------------------------------------------------
def run_sruthi_standardization_pipeline():
    # Step 1: Get audio file paths
    audio_path = get_audio_file_paths()
    # audio_path =['dataset/kaamavardhini/Enna_Gaanu.mp3','dataset/kaamavardhini/Ramanatham_Bhajeham.mp3']
    # audio_path = ['dataset/kaamavardhini/Sri_Gurupadanakha.mp3','dataset/kaamavardhini/Enna_Gaanu.mp3','dataset/kaamavardhini/Ramanatham_Bhajeham.mp3','dataset/kaamavardhini/Sami_Ni.mp3','dataset/kaamavardhini/Vaderadeivamu.mp3','dataset/kaamavardhini/Ninnu_Nera_Nammi.mp3']
    # audio_path = ['dataset/shankaraabharanam/Tukiya_Tiruvadi.mp3','dataset/shankaraabharanam/Sami_Ninne.mp3','dataset/shankaraabharanam/Dakshinamurte.mp3','dataset/shankaraabharanam/Undan_Paada_Pankayam.mp3','dataset/shankaraabharanam/Sri_Dakshinamurte.mp3']                        
    # audio_path = ['dataset/harikaambhoji/Saketha.mp3','dataset/harikaambhoji/Dinamani_Vamsha 2.mp3','dataset/harikaambhoji/Dinamanivamsa.mp3','dataset/harikaambhoji/Enadu_Manam.mp3','dataset/harikaambhoji/Dinamani_Vamsha.mp3']        
    # audio_path = ['dataset/kharaharapriya/Oru_murai_darishanam.mp3','dataset/kharaharapriya/Navasiddhi_Petralum.mp3','dataset/kharaharapriya/Sreenivasa_Thavacharanam.mp3','dataset/kharaharapriya/Sivanar_Manam.mp3','dataset/kharaharapriya/Satatam_Tavaka.mp3']
    # audio_path = ['dataset/maayamaalavagowlai/Vidulaku.mp3','dataset/maayamaalavagowlai/Deva_Deva_Kalayami.mp3','dataset/maayamaalavagowlai/Vidulaku_Mrokkeda.mp3','dataset/maayamaalavagowlai/Thulasidala.mp3','dataset/maayamaalavagowlai/Sri_Raja_Rajeswari.mp3']
    # audio_path = ['dataset/kalyani/Paarengum.mp3','dataset/kalyani/Vanajakshiro.mp3','dataset/kalyani/Isha_Paahimam.mp3','dataset/kalyani/Kanden_Kaliten.mp3','dataset/kalyani/Vanajakshi_-_Varnam.mp3']
    # audio_path = ['dataset/shanmugapriya/Mariveradikkevaraiya_Rama.mp3','dataset/shanmugapriya/Vilayada_Idu_Nerama.mp3','dataset/shanmugapriya/Kannanai_Pani_Maname.mp3','dataset/shanmugapriya/Kandanai_Nesithaal copy.mp3','dataset/shanmugapriya/Velan_Varuvaradi.mp3',]
    # audio_path = ['dataset/hanumathodi/Vazhi_Maraittirukkude.mp3','dataset/hanumathodi/Emani_Migula.mp3','dataset/hanumathodi/Varnam_Eranapai_Inta.mp3','dataset/hanumathodi/Karuna_Judavamma.mp3','dataset/hanumathodi/Tanigai_Valar.mp3']
    
    # Step 2: Get original sruthi information
    audio_arrays, sampling_rates, original_sruthi_frequency_hz, original_sruthi_frequency_midi, original_sruthi_note = get_sruthi_info_piptrack_pitchclass(audio_path)
    # Step 3: Standardize sruthi frequency
    audios_standardized_sruthi, std_sruthi_freq_hz, std_sruthi_freq_midi, std_sruthi_note = standardize_sruthi(audio_arrays, sampling_rates, original_sruthi_frequency_hz)
    # Step 4: Save standardized audio files
    # standardized_audio_paths = save_standardized_audios(audio_path, audios_standardized_sruthi, sampling_rates)
    # Step 5: Create base dataframe
    base_dataframe_std_sruthi = create_base_dataframe(audio_path, original_sruthi_note, original_sruthi_frequency_hz, original_sruthi_frequency_midi, std_sruthi_note, std_sruthi_freq_hz, std_sruthi_freq_midi, sampling_rates)
    # Step 6: Split audios into clips
    clipped_df = split_audio_into_clips(audio_path, audios_standardized_sruthi, sampling_rates, clip_duration_sec=30)
    # Step 7: Merge details to clipped dataframe
    clipped_df_std_sruthi_merged = merge_details_to_clip(clipped_df, base_dataframe_std_sruthi)
    
    print("sruthi_standardization_pipeline run successfully!!!")
    
    return clipped_df_std_sruthi_merged


#-------------------------------------------------------------------------
#Function to run the Raga Feature Extraction pipeline
#-------------------------------------------------------------------------
def run_raga_feature_extraction_pipeline(clipped_df_std_sruthi_merged):
    # Step 1: Pitch Contour extraction from clipped audios
    pitch_features_added_df = add_pitch_features_to_dataframe(clipped_df_std_sruthi_merged)
    # Step 2: Interval Sequnce extraction
    interval_features_added_df = add_interval_sequence_to_dataframe(pitch_features_added_df)
    # Step 3: First and Second order derivatives of f0 
    f0_derivative_and_second_derivative_added_df = add_f0_derivative_and_second_derivative_to_dataframe(interval_features_added_df)
    # Step 4: Extraction of Gamaka Descriptors
    gamaka_descriptors_added_df = add_gamaka_descriptors_to_dataframe(f0_derivative_and_second_derivative_added_df)
    # Step 5: Extraction of Mel Spectrograms
    mel_spectrogram_added_df = add_melspectrogram_to_dataframe(gamaka_descriptors_added_df)
    print("Midway till melspectrogram features file has been created successfully")
    # Step 6: Extraction of MFCC Features
    mfcc_features_added_df = add_mfcc_features_to_dataframe(mel_spectrogram_added_df)
    # Step 7: Extraction of Chroma Features
    chroma_features_added_df = add_chroma_features_to_dataframe(mfcc_features_added_df)
    
    raga_features_merged_df = chroma_features_added_df.copy()
    
    return raga_features_merged_df
