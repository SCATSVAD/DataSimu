import torch
import numpy as np
import os
import glob
import pandas as pd
import torchaudio
def find_matching_indices(tensor_list, array_list):
    matching_indices = []
    for i, tensor in enumerate(tensor_list):
        for j, array in enumerate(array_list):
            if torch.norm(tensor - torch.from_numpy(array)) < 0.5:
                matching_indices.append(i)
    return list(set(matching_indices))

def process_librispeech(root_dir, save_dir):
    records = []

    # 遍历所有子文件夹和文件
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"): 
                text_path = os.path.join(subdir, file)
                with open(text_path, 'r') as text_file:
                    for line in text_file:
                        parts = line.strip().split()
                        audio_file_name = parts[0] + ".flac" 
                        transcript = ' '.join(parts[1:])
                        records.append([audio_file_name, transcript])

    df = pd.DataFrame(records, columns=['name', 'words'])
    df.to_csv(save_dir, index=False)

def load_transcripts(csv_file_path):
    df = pd.read_csv(csv_file_path)
    transcripts_dict = pd.Series(df.words.values,index=df.name).to_dict()

    return transcripts_dict

def load_reco2dur(path):
    reco2dur = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.strip('\n').split('\t')
            reco = data[0]
            time = data[1]
            reco2dur[reco] = float(time)
            
    return reco2dur

def load_reco2segments(path):
    reco2segments = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            reco = line[1]
            if reco not in reco2segments:
                reco2segments[reco] = []
            reco2segments[reco].append([float(line[2]), float(line[3])])
        
    return reco2segments

def load_reco2num_spk(path):
    reco2num_spk = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.strip('\n').split()
            reco2num_spk[data[0]] = float(data[1])
            
    return reco2num_spk

def load_rttm(path):
    rttm_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if len(line.split()) == 10:
                _, reco, channel, st, dt, _, _, spk_id, _, _ = line.split()
            else:
                _, reco, channel, st, dt, _, _, spk_id, _    = line.split()
            rttm_data = dict(reco=reco, st=float(st), dt=float(dt), spk_id=spk_id)
            rttm_list.append(rttm_data)
            
    return rttm_list

def init_output_dir(output_path):
    target_dir = os.path.split(output_path)[0]
    if len(target_dir) > 0 and (not os.path.exists(target_dir)):
        os.makedirs(target_dir, exist_ok=True)
  
    return  

def save_reco2dur(path, reco2dur):
    init_output_dir(path)
    with open(path, 'w') as f:
        for reco, dur in reco2dur.items():
            line = reco + '\t' + str(dur) + '\n'
            f.writelines(line)

def save_spk2dur(path, spk2dur):
    init_output_dir(path)
    with open(path, 'w') as f:
        for reco, dur in spk2dur.items():
            line = reco + '\t' + str(dur) + '\n'
            f.writelines(line)

def process_rttm_to_spk2dur(rttm_path):
    """
    Process an RTTM file to generate a dictionary mapping each speaker to their longest cumulative speaking duration across recordings.

    Args:
        rttm_path (str): Path to the RTTM file.

    Returns:
        dict: A dictionary where keys are speaker IDs and values are their longest cumulative speaking durations (float).
    """
    spk2dur = {}

    with open(rttm_path, 'r') as file:
        for line in file:
            # Split the line into fields
            parts = line.strip().split()

            # Ensure the line follows the expected RTTM format
            if len(parts) >= 9 and parts[0] == "SPEAKER":
                reco_id = parts[1]  # Extract the recording ID
                spk_id = parts[7]  # Extract the speaker ID
                dur = float(parts[4])  # Extract the duration

                # Initialize nested dictionary for the speaker if not present
                if spk_id not in spk2dur:
                    spk2dur[spk_id] = {}

                # Accumulate duration for this recording ID
                if reco_id in spk2dur[spk_id]:
                    spk2dur[spk_id][reco_id] += dur
                else:
                    spk2dur[spk_id][reco_id] = dur
                
    
    # Find the longest cumulative duration across recordings for each speaker
    spk2dur_max = {spk_id: max(durations.values()) for spk_id, durations in spk2dur.items()}
    return spk2dur_max
def process_rttm_to_reco2spkdur(rttm_path):
    """
    Process an RTTM file to generate a dictionary mapping each speaker to their longest cumulative speaking duration across recordings.

    Args:
        rttm_path (str): Path to the RTTM file.

    Returns:
        dict: A dictionary where keys are speaker IDs and values are their longest cumulative speaking durations (float).
    """
    reco2spkdur = {}

    with open(rttm_path, 'r') as file:
        for line in file:
            # Split the line into fields
            parts = line.strip().split()

            # Ensure the line follows the expected RTTM format
            if len(parts) >= 9 and parts[0] == "SPEAKER":
                reco_id = parts[1]  # Extract the recording ID
                spk_id = parts[7]  # Extract the speaker ID
                dur = float(parts[4])  # Extract the duration

                # Initialize nested dictionary for the speaker if not present
                if reco_id not in reco2spkdur:
                    reco2spkdur[reco_id] = {}

                # Accumulate duration for this recording ID
                if spk_id in reco2spkdur[reco_id]:
                    reco2spkdur[reco_id][spk_id] += dur
                else:
                    reco2spkdur[reco_id][spk_id] = dur
                
    return reco2spkdur

def get_wav_length(file_path):
    # 读取音频文件
    waveform, sample_rate = torchaudio.load(file_path)
    # 计算音频长度（秒）
    duration = waveform.size(1) / sample_rate
    return duration

def get_spk2dur_from_libri():
    with open(f"data/train_speech.txt", "r") as f:
        speech_list = f.readlines()
    save_path = 'data/train_libri_spk2dur'
    speech_list = [speech.strip() for speech in speech_list]
    reco2dur = {}
    for idx, path in enumerate(speech_list):
        length = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.flac'):
                    file_path = os.path.join(root, file)
                    length += get_wav_length(file_path)
        reco2dur[path] = length
        print(f'Processing {idx}/{len(speech_list)}')


def find_dirs_with_json(root_dir):
    """
    查找 `root_dir` 下 **第二层目录** 是否包含 JSON 文件，若有，则保存 **第一层目录** 的名称。

    Args:
        root_dir (str): 根目录路径

    Returns:
        list: 包含 JSON 文件的第一层目录名
    """
    valid_dirs = []

    # 遍历 root_dir 下的所有第一层目录
    for first_level_dir in os.listdir(root_dir):
        first_level_path = os.path.join(root_dir, first_level_dir)

        # 使用 glob 查找 JSON 文件
        json_files = glob.glob(os.path.join(first_level_path, "*.json"))

        # 如果存在 JSON 文件，则记录第一层目录名
        if json_files:
            valid_dirs.append(first_level_dir)

    return valid_dirs


def remove_silent_samples(waveform: torch.Tensor) -> torch.Tensor:
    """
    移除音频信号中所有取值为 0 的采样点。
    
    参数：
        waveform (torch.Tensor): 输入音频信号，形状为 (1, num_samples) 或 (channels, num_samples)
    
    返回：
        torch.Tensor: 处理后的音频信号，移除了所有取值为 0 的采样点。
    """
    if waveform.dim() != 2:
        raise ValueError("输入的 waveform 需要是二维的 (channels, num_samples)")
    
    # 找到所有非零采样点
    nonzero_indices = torch.any(waveform != 0, dim=0)
    
    # 仅保留非零采样点
    cleaned_waveform = waveform[:, nonzero_indices]
    
    return cleaned_waveform