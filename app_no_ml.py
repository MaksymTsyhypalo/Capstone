import numpy as np
import librosa
import scipy.fft as fft
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
import math

TARGET_SR = 16000
TARGET_SECONDS = 1.0
TARGET_LENGTH = int(TARGET_SR * TARGET_SECONDS)
SEGMENT_STEP = int(TARGET_SR * 0.5)

SPEED_OF_SOUND = 343.0  # m/s (approximate)
ARRAY_RADIUS = 0.1
IMAGE_SIZE = 224        # Kept for Gradio output placeholder consistency

#microphone array
MIC_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)] 
NUM_MICS = 8

# Calculate 8 microphone positions for a UCA on the XY plane
MIC_POSITIONS = []
for i in range(NUM_MICS):
    # Angle for mic i (degrees to radians)
    angle_rad = np.deg2rad(i * 360 / NUM_MICS) 
    x = ARRAY_RADIUS * math.cos(angle_rad)
    y = ARRAY_RADIUS * math.sin(angle_rad)
    MIC_POSITIONS.append(np.array([x, y, 0])) # [x, y, z]

# Calculate the maximum possible TDoA based on array size
MAX_DELAY_SECONDS = (2 * ARRAY_RADIUS) / SPEED_OF_SOUND
MAX_DELAY_SAMPLES = int(MAX_DELAY_SECONDS * TARGET_SR)


# --- 2. GCC-PHAT VECTOR FUNCTION (Modified from original) ---

def calculate_gcc_phat_vector(sig1: np.ndarray, sig2: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the full GCC-PHAT correlation vector and the corresponding delay indices.
    """
    min_len = min(len(sig1), len(sig2))
    sig1 = sig1[:min_len]
    sig2 = sig2[:min_len]
    
    # Use a power-of-2 FFT length for efficiency
    n_fft = int(2**np.ceil(np.log2(len(sig1))))
    
    SIG1 = fft.rfft(sig1, n_fft)
    SIG2 = fft.rfft(sig2, n_fft)
    R12 = SIG1 * np.conj(SIG2)
    R12_phat = R12 / (np.abs(R12) + 1e-6)
    r = fft.irfft(R12_phat, n_fft)
    
    # Shift the vector so zero delay is at the start (standard FFT output)
    
    # Shift to center TDoA=0 (index 0 for IRFFT is always TDoA=0)
    
    # Create the time-domain vector with zero delay at index 0
    # Then shift it to center the zero delay, making positive/negative delays symmetric
    r_shifted = np.roll(r, -len(r) // 2)
    
    # Extract the physically relevant central window
    center_idx = len(r_shifted) // 2
    
    # Ensure the slice is within bounds
    start_idx = center_idx - MAX_DELAY_SAMPLES
    end_idx = center_idx + MAX_DELAY_SAMPLES + 1
    
    feature_vector = r_shifted[start_idx:end_idx]
    
    # Create corresponding delay indices (in samples)
    delay_indices = np.arange(-MAX_DELAY_SAMPLES, MAX_DELAY_SAMPLES + 1)
    
    # Normalize the vector (optional but helpful for stability)
    feature_vector = (feature_vector - np.min(feature_vector)) / (np.max(feature_vector) - np.min(feature_vector) + 1e-6)

    return feature_vector, delay_indices


# 3D SRP-PHAT Solver

def _process_segment_srp_phat(segment_signals: List[np.ndarray], sr: int) -> Tuple[float, float, np.ndarray]:
    """Processes a single 1.0s segment using the SRP-PHAT algorithm."""
    
    #All GCC-PHAT vectors (1-D correlation)
    gcc_vectors = {}
    gcc_delays = None
    for mic1, mic2 in MIC_PAIRS:
        gcc_vec, gcc_delays = calculate_gcc_phat_vector(segment_signals[mic1], segment_signals[mic2], sr)
        gcc_vectors[(mic1, mic2)] = gcc_vec
        
    # Search grid (step 5 degrees)
    azimuth_grid = np.deg2rad(np.arange(-180, 180, 5)) 
    elevation_grid = np.deg2rad(np.arange(-90, 95, 5)) 
    
    max_power = -np.inf
    best_azim_rad = 0.0
    best_elev_rad = 0.0
    
    # 3. Iterate through the grid and calculate power 
    for elev_rad in elevation_grid:
        # Optimization: pre-calculate cos(elev) and sin(elev)
        cos_e = math.cos(elev_rad)
        sin_e = math.sin(elev_rad)

        for azim_rad in azimuth_grid:
            
            # Calculate unit vector u (pointing from source to origin)
            # This is the direction vector of the sound wave.
            u = np.array([
                cos_e * math.cos(azim_rad),
                cos_e * math.sin(azim_rad),
                sin_e
            ])
            
            current_power = 0.0
            
            # Sum the GCC-PHAT response for all pairs
            for mic1, mic2 in MIC_PAIRS:
                P1, P2 = MIC_POSITIONS[mic1], MIC_POSITIONS[mic2]
                
                # Theoretical TDoA in seconds
                tau_theo = (1 / SPEED_OF_SOUND) * np.dot(u, P2 - P1)
                
                # Theoretical TDoA in samples
                tau_theo_samples = int(np.round(tau_theo * sr))
                
                # Find the index corresponding to the theoretical delay
                try:
                    # Look up the index in the pre-calculated delay array
                    idx = np.where(gcc_delays == tau_theo_samples)[0][0]
                    
                    # Accumulate the GCC-PHAT value at this delay (Steered Response Power)
                    current_power += gcc_vectors[(mic1, mic2)][idx]
                except IndexError:
                    # Theoretical delay is outside the max physical limit (shouldn't happen 
                    # if MAX_DELAY_SAMPLES is set correctly, but handled safely)
                    pass
            
            # 4. Find the angle with the maximum power
            if current_power > max_power:
                max_power = current_power
                best_azim_rad = azim_rad
                best_elev_rad = elev_rad
                
    # 5. Convert best angle to degrees and clip to plot ranges
    predicted_azimuth_deg = np.rad2deg(best_azim_rad)
    predicted_elevation_deg = np.rad2deg(best_elev_rad)
    
    # Normalize azimuth to the desired [-90, 90] range for plotting
    predicted_azimuth_deg = predicted_azimuth_deg % 360
    if predicted_azimuth_deg > 180: predicted_azimuth_deg -= 360
    predicted_azimuth_deg = np.clip(predicted_azimuth_deg, -90.0, 90.0)

    # Normalize elevation to the desired [-15, 75] range for plotting
    predicted_elevation_deg = np.clip(predicted_elevation_deg, -15.0, 75.0)

    #Prepare Feature Map visualization (using the final angle's "power map")
    gcc_vis = gcc_vectors[(0, 4)]
    # Resize the 1D GCC-PHAT vector to a 2D square image (e.g., repeating rows)
    gcc_vis_2d = np.tile(gcc_vis, (IMAGE_SIZE, 1))
    
    # Scale feature_vector length to IMAGE_SIZE (224) width
    if gcc_vis_2d.shape[1] != IMAGE_SIZE:
        # Simple upsampling by repeating columns
        scale_factor = IMAGE_SIZE / gcc_vis_2d.shape[1]
        indices = np.round(np.arange(IMAGE_SIZE) / scale_factor).astype(int)
        indices[indices >= gcc_vis_2d.shape[1]] = gcc_vis_2d.shape[1] - 1
        feature_vis = gcc_vis_2d[:, indices]
    else:
        feature_vis = gcc_vis_2d

    feature_vis = (feature_vis * 255).astype(np.uint8)

    return predicted_azimuth_deg, predicted_elevation_deg, feature_vis

# Trajectory Plotting

def plot_trajectory(times: List[float], azimuths: List[float], elevations: List[float]):
    """Generates a Matplotlib plot showing predicted azimuth and elevation over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    if not azimuths:
        fig.suptitle("No Data Processed", fontsize=16)
        ax1.text(0.5, 0.5, "Upload an 8-channel file longer than 1 second.", ha='center', va='center', transform=ax1.transAxes, color='gray')
        ax2.axis('off')
    else:
        fig.suptitle("Predicted Direction of Arrival (DoA) Trajectory (SRP-PHAT)", fontsize=16)

        # Azimuth Plot (Top)
        ax1.plot(times, azimuths, marker='o', linestyle='-', color='#FF4500', linewidth=2, markersize=4, label='Azimuth')
        ax1.set_ylabel("Azimuth (Degrees, -90° to 90°)")
        ax1.set_yticks(np.arange(-90, 91, 30)) 
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylim(-90, 90)

        # Elevation Plot (Bottom)
        ax2.plot(times, elevations, marker='s', linestyle='--', color='#1E90FF', linewidth=2, markersize=4, label='Elevation')
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Elevation (Degrees, -15° to 75°)") 
        ax2.set_yticks(np.arange(-15, 76, 15)) 
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(-15, 75)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.close(fig)
    return fig


# Main Prediction Function

def predict(audio: Union[str, None]):
    """Loads multi-channel audio, slices it, predicts DoA (Azimuth & Elevation) for each using SRP-PHAT."""
    
    empty_feature_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if audio is None:
        return "Waiting for audio input...", plot_trajectory([], [], []), empty_feature_map

    try:
        waveform_np, sr = librosa.load(audio, sr=TARGET_SR, mono=False)
    except Exception as e:
        return f"Error loading audio file: {e}", plot_trajectory([], [], []), empty_feature_map

    n_channels = waveform_np.shape[0] if waveform_np.ndim >= 2 else 1
    total_samples = waveform_np.shape[-1]
    
    # Validation checks
    if n_channels < NUM_MICS:
        status_message = (
            f"Localization Unavailable: Found {n_channels} channel(s). "
            f"**{NUM_MICS} channels required** for the circular array DoA model."
        )
        return status_message, plot_trajectory([], [], []), empty_feature_map

    if total_samples < TARGET_LENGTH:
        status_message = f"Localization Unavailable: Audio too short (need {TARGET_SECONDS}s)."
        return status_message, plot_trajectory([], [], []), empty_feature_map

    # Slicing and Prediction Loop
    azimuths = []
    elevations = []
    times = []
    last_feature_vis = empty_feature_map

    for start_sample in range(0, total_samples - TARGET_LENGTH + 1, SEGMENT_STEP):
        end_sample = start_sample + TARGET_LENGTH
        segment_np = waveform_np[:, start_sample:end_sample]
        segment_signals = [segment_np[i, :] for i in range(n_channels)]
        
        # Predict angle for this segment using SRP-PHAT
        predicted_azimuth, predicted_elevation, feature_vis = _process_segment_srp_phat(segment_signals, sr)

        # Store results
        current_time = (start_sample + TARGET_LENGTH / 2) / sr
        
        azimuths.append(predicted_azimuth)
        elevations.append(predicted_elevation)
        times.append(current_time)
        last_feature_vis = feature_vis

    # Generate Trajectory Plot
    trajectory_plot = plot_trajectory(times, azimuths, elevations)

    status_message = (
        f"3D Localization Trajectory Calculated: Processed {len(times)} segments "
        f"using SRP-PHAT on array geometry (Radius: {ARRAY_RADIUS*100:.1f} cm)."
    )
    
    return status_message, trajectory_plot, last_feature_vis