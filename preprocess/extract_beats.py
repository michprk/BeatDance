import copy
import cv2
import os
import ffmpeg
import torch
import librosa
import librosa.display

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter as G

# ========== FFmpeg Video Writer ==========
class Writer():
    def __init__(self, output_file, input_fps, input_framesize):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.ff_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt="bgr24",
                   s=f'{input_framesize[1]}x{input_framesize[0]}', r=input_fps)
            .output(output_file, pix_fmt="yuv420p", vcodec="libx264",
                    format="mp4", video_bitrate="500k")
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )

    def __call__(self, frame):
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()

# ========== FFProbe Setup ==========
def get_video_duration(video_file):
    metadata = ffmpeg.probe(video_file)
    return float(metadata["format"]["duration"])

def get_video_fps(video_file):
    metadata = ffmpeg.probe(video_file)
    for stream in metadata["streams"]:
        if stream["codec_type"] == "video":
            return eval(stream["avg_frame_rate"])
    return 30

# ========== OpenPose Setup ==========
from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')

# ========== Dance Beat Calculation ==========
def calc_db(keypoints, total_frames):
    keypoints = np.nan_to_num(np.array([kp[:18] if len(kp) >= 18 else np.vstack([kp, np.full((18 - len(kp), 3), np.nan)]) for kp in keypoints]), nan=0.0)
    kinetic_vel = np.zeros(total_frames)
    if keypoints.shape[0] > 1:
        velocity = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
        valid_len = min(len(velocity), len(kinetic_vel) - 1)
        kinetic_vel[1:valid_len + 1] = np.nan_to_num(velocity[:valid_len], nan=0.0)
    max_val = np.max(kinetic_vel)
    if max_val > 0:
        kinetic_vel /= max_val
    kinetic_vel = np.nan_to_num(G(kinetic_vel, sigma=2))
    motion_beats = (argrelextrema(kinetic_vel, np.less)[0][argrelextrema(kinetic_vel, np.less)[0] < total_frames],)
    return motion_beats, kinetic_vel

def ensure_audio_matches_video(audio_file, video_file):
    """ Ensure audio length matches video duration exactly """
    video_duration = get_video_duration(video_file)
    y, sr = librosa.load(audio_file, sr=None)
    audio_duration = len(y) / sr  # Compute audio length in seconds
    print(f"Video duration: {video_duration}s, Audio duration: {audio_duration}s")
    if audio_duration > video_duration:
        print(f"Trimming audio from {audio_duration}s to {video_duration}s")
        y = y[:int(video_duration * sr)]
    elif audio_duration < video_duration:
        print(f"Padding audio from {audio_duration}s to {video_duration}s")
        y = np.pad(y, (0, int((video_duration - audio_duration) * sr)), 'constant')
    temp_audio_file = "temp_synced_audio.wav"
    sf.write(temp_audio_file, y, sr)
    return temp_audio_file, sr, video_duration

def extract_music_beats(audio_file, fps, total_frames, audio_folder, output_music_folder):
    y, sr = librosa.load(audio_file, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    music_beats = np.round(beats * fps).astype(int)
    music_beats = music_beats[music_beats < total_frames]
    torch.save(torch.tensor(music_beats), os.path.join(output_music_folder, os.path.basename(audio_folder).replace(".wav", ".pt")))
    return music_beats

def plot_motion_vs_music_beats(kinetic_vel, motion_beats, music_beats, total_frames, plot_output_file):
    plt.figure(figsize=(10, 5))
    plt.plot(range(total_frames), kinetic_vel, label="Kinetic Velocity", color="blue")
    plt.scatter(motion_beats[0], kinetic_vel[motion_beats[0]], color="red", label="Motion Beats", zorder=3)
    if len(music_beats) > 0:
        plt.vlines(music_beats, ymin=0, ymax=np.max(kinetic_vel), colors="green", linestyle="dashed", alpha=0.6, label="Music Beat")
    plt.xlabel("Frame")
    plt.ylabel("Velocity")
    plt.title("Motion vs. Music Beats")
    plt.legend()
    plt.grid()
    plt.xlim(0, total_frames)
    plt.ylim(0, np.max(kinetic_vel) + 0.05)
    plt.savefig(plot_output_file)

def process_video(video_file, output_folder):
    video_fps = get_video_fps(video_file)
    video_duration = get_video_duration(video_file)
    total_frames = int(video_fps * video_duration)
    cap = cv2.VideoCapture(video_file)
    keypoints_list = []
    output_video_file = os.path.join(output_folder, os.path.basename(video_file))
    writer = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        candidate, subset = body_estimation(frame)
        posed_frame = util.draw_bodypose(copy.deepcopy(frame), candidate, subset)
        if len(candidate) > 0:
            keypoints_list.append(candidate[:, :3])
        if writer is None:
            writer = Writer(output_video_file, video_fps, posed_frame.shape[:2])
        writer(copy.deepcopy(posed_frame))
    cap.release()
    writer.close()
    motion_beats, kinetic_vel = calc_db(keypoints_list, total_frames)
    torch.save(torch.tensor(motion_beats[0]), os.path.join(output_folder, os.path.basename(video_file).replace(".mp4", ".pt")))
    return motion_beats, kinetic_vel, total_frames

def process_files():
    music_folder = r"/home/sangheon/Desktop/BeatDance/data/dance_music"
    video_folder = r"/home/van/data/SonyDanceSegmentedVideo_compress10fps"
    output_music_folder = r"/home/van/scripts/BeatDance/data/dance_video/music_beat"
    output_video_folder = r"/home/van/scripts/BeatDance/data/dance_video/video_beat"
    os.makedirs(output_music_folder, exist_ok=True)
    os.makedirs(output_video_folder, exist_ok=True)
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            motion_beats, kinetic_vel, total_frames = process_video(video_path, output_video_folder)
            video_base_name = video_file.replace(".mp4", "").replace("_clip_", "_audio_")
            audio_path = os.path.join(music_folder, f"{video_base_name}.wav")
            plot_output_file = os.path.join(output_video_folder, video_file.replace(".mp4", ".png"))
            # print(audio_path)
            if os.path.exists(audio_path):
                synced_audio, sr, _ = ensure_audio_matches_video(audio_path, video_path)
                music_beats = extract_music_beats(synced_audio, get_video_fps(video_path), total_frames, audio_path, output_music_folder)
                plot_motion_vs_music_beats(kinetic_vel, motion_beats, music_beats, total_frames, plot_output_file)
                print(f"Processed {video_file}: {len(music_beats)} music beats, {len(motion_beats[0])} motion beats")
            else:
                print(f"No matching audio for {video_file}")
if __name__ == "__main__":
    process_files()