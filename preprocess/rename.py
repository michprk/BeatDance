import os

def rename_files(folder_path):
    """
    Renames files in the specified folder by replacing '_audio_' with '_clip_' in filenames.

    Args:
        folder_path (str): The path to the folder containing the files.
    """
    for filename in os.listdir(folder_path):
        if '_audio_' in filename and filename.endswith('.pt'):
            new_filename = filename.replace('_audio_', '_clip_')
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")

# 🔹 Set the target folder path (Change this before running)
folder_path = r"/home/van/scripts/BeatDance/data/dance_video/music_feature"  # Replace with actual folder path

# Run the renaming function
rename_files(folder_path)

# import os

# def rename_files(folder_path):
#     """
#     Renames files in the specified folder by removing '_features' from filenames.

#     Args:
#         folder_path (str): The path to the folder containing the files.
#     """
#     for filename in os.listdir(folder_path):
#         if '_features.pt' in filename:
#             new_filename = filename.replace('_features.pt', '.pt')
#             old_path = os.path.join(folder_path, filename)
#             new_path = os.path.join(folder_path, new_filename)

#             os.rename(old_path, new_path)
#             print(f"Renamed: {filename} → {new_filename}")

# # 🔹 Set the target folder path (Change this before running)
# folder_path = r"/home/van/scripts/BeatDance/data/dance_video/video_feature"  # Replace with actual folder path

# # Run the renaming function
# rename_files(folder_path)