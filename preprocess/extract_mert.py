import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# Define learnable weighted aggregation layer
class WeightedAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)

    def forward(self, hidden_states):
        return self.aggregator(hidden_states).squeeze(1)  # Output: [10, 768]

def extract_mert_features(audio_path, model, processor, aggregator):
    """
    Extracts MERT features from a 10-second audio file.
    The extracted feature will have a shape of [10, 768].
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if audio is stereo
    if waveform.shape[0] > 1:  # If there are multiple channels
        print(f"Converting stereo to mono for: {audio_path}")
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono by averaging channels

    # Resample if needed
    resample_rate = processor.sampling_rate
    if sample_rate != resample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=resample_rate)
        waveform = resampler(waveform)

    # Convert to expected format (float32)
    input_audio = waveform.squeeze().float()

    # Process audio and run model
    inputs = processor(input_audio.numpy(), sampling_rate=resample_rate, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()  # [13, Time Steps, 768]

    # Set fixed L = 10 intervals
    total_time_steps = all_layer_hidden_states.shape[1]  # Total frames in 10s
    interval_frames = total_time_steps // 10  # Divide into 10 equal segments

    print(f"Total frames: {total_time_steps}, Frames per interval: {interval_frames}, L = 10")

    # Segment into 10 intervals and compute average feature per interval
    interval_features = []
    for i in range(10):
        start_idx = i * interval_frames
        end_idx = min((i + 1) * interval_frames, total_time_steps)

        # Average over time steps within each interval
        interval_avg = all_layer_hidden_states[:, start_idx:end_idx, :].mean(dim=1)  # [13, 768]
        interval_features.append(interval_avg)

    interval_features = torch.stack(interval_features)  # Shape: [10, 13, 768]
    interval_features = interval_features.permute(2, 1, 0)  # Shape: [768, 13, 10]
    weighted_avg_hidden_states = aggregator(interval_features)  # Shape: [768, 10]
    weighted_avg_hidden_states = weighted_avg_hidden_states.permute(1, 0)  # Shape: [10, 768]
    print(f"Final shape after Conv1d: {weighted_avg_hidden_states.shape}")  # Expected [10, 768]

    return weighted_avg_hidden_states

def process_music_folder(music_folder, output_folder):
    """
    Processes all .wav files in the music folder and saves extracted features.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load model and processor
    print("Loading MERT model...")
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    aggregator = WeightedAggregator()

    # Process all .wav files
    for filename in os.listdir(music_folder):
        if filename.endswith(".wav"):
            audio_path = os.path.join(music_folder, filename)
            print(f"Processing: {audio_path}")

            try:
                # Extract features
                features = extract_mert_features(audio_path, model, processor, aggregator)

                # Save features
                output_path = os.path.join(output_folder, filename.replace(".wav", ".pt"))
                torch.save(features, output_path)
                print(f"Saved features to {output_path}")

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

    print("Feature extraction complete!")


def main():
    """
    Main function to set paths and initiate processing.
    """
    music_folder = r"/home/van/data/SonyDanceSegmentedAudio"  # Change this to your input folder
    output_folder = r"/home/van/scripts/BeatDance/data/music_feature"  # Change this to your output folder

    process_music_folder(music_folder, output_folder)

if __name__ == "__main__":
    main()