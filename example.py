import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torchaudio
import numpy as np
import gc, glob
from rich import print
import time
import math
import random
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import habitat_sim.sim
import SonicSim_rir
import SonicSim_habitat
import SonicSim_audio
import SonicSim_moving
import tool_utils
import multiprocessing
import logging


def generate_all_channels_for_a_wav(scene, sample_rate, results_dir, noise_path, music_path, source_spks, rttm, reco, length):
    sample_rate = sample_rate

    spk_num = len(source_spks)
    room = scene
    scene = SonicSim_rir.Scene(
        room,
        [None],  # placeholder for source class
        include_visual_sensor=False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    
    spks_nav_points = [SonicSim_rir.get_nav_idx(scene, distance_threshold=5.0) for _ in range(spk_num)]
    # Generate the source positions
    spks_nav_mid_points = [spks_nav_points[i][len(spks_nav_points[i]) // 2] for i in range(spk_num)]
    # Generate the music and noise positions
    noise_music_points = SonicSim_rir.get_nav_point_from_grid_points(scene, spks_nav_mid_points, distance_threshold=6.0, num_points=2)
    # Generate the microphone position
    mic_points = SonicSim_rir.get_nav_point_from_grid_points(scene, spks_nav_mid_points, distance_threshold=6.0, num_points=1)[0]

    # prepare params for source audio generation
    ali_spk2source_path = []
    spks = list(set(item['spk_id'] for item in rttm))
    print_source = []
    for i in range(spk_num):
        spk2path = {}
        spk2path[spks[i]] = source_spks[i]
        print_source.append(source_spks[i])
        ali_spk2source_path.append(spk2path)


    # Get noise and music audio
    noise_audio, noise_start_end, noise_audioname = SonicSim_audio.create_background_audio(noise_path, length)
    music_audio, music_start_end, music_audioname = SonicSim_audio.create_background_audio(music_path, length)
    # Get source audios 
    source_audios = []
    source_dir = f'{results_dir}/{reco}_source'
    os.makedirs(source_dir, exist_ok=True)
    for i in range(spk_num):
        source_audio = SonicSim_audio.create_ali_audio(ali_spk2source_path[i], length, rttm)
        source_audio_dir = f'{source_dir}/speaker_{i}.wav'
        if not os.path.exists(source_audio_dir):
            torchaudio.save(source_audio_dir, source_audio, sample_rate=sample_rate)
        source_audios.append(source_audio)
    
    def mix_ir_source(channel_type,mic_array_list,mic_config,rand_dist=None):
            
        # Generate RIRs
        output_dir = f'{results_dir}/{reco}_{mic_config}'
        if os.path.exists(output_dir):
            print(f'{output_dir} is existed')
            return
        os.makedirs(output_dir)
        ir_save_dir = f'{output_dir}/rir_save_{mic_config}.pt'

        IRI_outputs = []
        for i in range(len(spks_nav_mid_points)):
            ir_output = SonicSim_audio.generate_rir_combination(
                        room, [spks_nav_mid_points[i]], [mic_points], [90], mic_array_list, channel_type
                )
            IRI_outputs.append(ir_output)
            del ir_output
            gc.collect()
        torch.save(IRI_outputs, ir_save_dir)
        audios = []  # Collect all audio mixed with IR and mix to obtain overlapping speech
        for i in range(spk_num):
            receiver_audio = SonicSim_moving.convolve_fixed_receiver(source_audios[i], IRI_outputs[i])
            # (channels, time)
            audios.append(torch.from_numpy(receiver_audio))
        # Get rir for noise and music
        if channel_type == 'CustomArrayIR':
            rir_noise = SonicSim_rir.create_custom_arrayir(room, noise_music_points[0], mic_points, mic_array=mic_array_list, filename=None, receiver_rotation=90, channel_order=0)
            rir_music = SonicSim_rir.create_custom_arrayir(room, noise_music_points[1], mic_points, mic_array=mic_array_list, filename=None, receiver_rotation=90, channel_order=0)
        else:
            rir_noise = SonicSim_rir.render_ir(room, noise_music_points[0], mic_points, filename=None, receiver_rotation=90, channel_type=channel_type, channel_order=0)
            rir_music = SonicSim_rir.render_ir(room, noise_music_points[1], mic_points, filename=None, receiver_rotation=90, channel_type=channel_type, channel_order=0)

        rir_noise = torch.from_numpy(SonicSim_moving.convolve_fixed_receiver(noise_audio, rir_noise.cpu()))
        rir_music = torch.from_numpy(SonicSim_moving.convolve_fixed_receiver(music_audio, rir_music.cpu()))
        audios.append(rir_noise)
        audios.append(rir_music)
        mixed_audio = torch.stack(audios).sum(dim=0)
        torchaudio.save(f'{output_dir}/mixed_audio.wav', mixed_audio, sample_rate=sample_rate)
        
        json_dicts = {
            'reco': reco,
            'length': length,
            'mic_config': mic_config,
            'distance': rand_dist,
            'room': room,
            'spk_num': len(print_source),
            'source': print_source,
            'noise': {
                'audio': noise_audioname,
                'start_end_points': noise_start_end
            },
            'music': {
                'audio': music_audioname,
                'start_end_points': music_start_end
            },
        }
        with open(f'{output_dir}/json_data.json', 'w') as f:
            logging.info(json_dicts)
            json.dump(json_dicts, f)
        print(f'The {mic_config} channel(s) audio of {reco} has been generated')
    
    for c in [1,2]:
        channel_type, mic_array_list = get_more_type_array(c, False)
        mic_config = channel_type  # Mono/Binaural
        mix_ir_source(channel_type, mic_array_list, mic_config)

    for c in [4,6,8]:
        for isLinear in [True, False]:
            channel_type, mic_array_list, rand_dist  = get_more_type_array(c,isLinear=isLinear)
            array = 'linear' if isLinear else 'circular'
            mic_config =  str(c) + array #4linear
            mix_ir_source(channel_type, mic_array_list, mic_config, rand_dist)




def generate_circular_mic_array(count, diameter):
    '''
    Generate coordinates for a circular microphone array.
    Args:
        count: the number of microphones
        diameter: the diameter of circular mic arrays, unit is meter
    Returns:
        List of (x, y, z) coordinates for each microphone.
    '''

    radius = diameter / 2

    mic_array_list = []
    for i in range(count):
        angle = i * 2 * math.pi / count
        x = round(radius * math.cos(angle), 3)
        y = 0
        z = round(radius * math.sin(angle), 3)
        mic_array_list.append([x,y,z])

    return mic_array_list

def generate_linear_mic_array(count, distance):
    '''
    Generate coordinates for a linear microphone array with the distance between mics.
    distance is measured in meter
    '''
    mic_array_list = []
    z = -distance
    for _ in range(count):
        z += distance
        mic_array_list.append([0, 0, z])
    return mic_array_list

def get_more_type_array(count, isLinear=False, diameter=0.102, distance=0.04):
    '''
    the default diameter 0.102 is the same as alimeeting
    return the channel type and mic array list when mic count is given
    # For CustomArrayIR
    channel_type = 'CustomArrayIR'
    mic_array_list = [
        [0, 0, 0],
        [0, 0, 0.04],
        [0, 0, 0.08],
        [0, 0, 0.12]
    ] # 4-channel linear microphone array
    channel_type = 'CustomArrayIR'
    mic_array_list = [
        [0, 0, -0.035],
        [0.035, 0, 0],
        [0, 0, 0.035],
        [-0.035, 0, 0]
    ] # 4-channel circular microphone array
    '''
    diameter = round(random.uniform(0.082, 0.122), 3)  # 102mm 前后 20
    distance = round(random.uniform(0.02, 0.06), 2)  # 40mm 前后 20

    assert count <= 8, 'The count of channels should not more than 8'
    if count == 1:
        return 'Mono', None
    elif count == 2:
        return 'Binaural', None
    elif count == 4 or count == 6 or count == 8:
        if isLinear:
            return 'CustomArrayIR', generate_linear_mic_array(count, distance), distance
        else:
            return 'CustomArrayIR', generate_circular_mic_array(count, diameter), diameter
    else:
        raise ValueError(f'{count} must be [1,2,4,6,8]')

def remove_spks_and_scenes(root, recos):
    exist_spks = []
    exist_scenes = []
    for reco in recos:
        path = os.path.join(root, reco)
        json_files = glob.glob(os.path.join(path, "*.json"))
        if len(json_files) == 0:
            continue
        json_file = json_files[0]
        with open(json_file, 'r')as f:
            data = json.load(f)
            exist_spks += data['source']
            exist_scenes.append(data['room'])
    return exist_spks, exist_scenes

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(f'SonicSet-train.log'), logging.StreamHandler()])
    
    multiprocessing.set_start_method("spawn")
    # Supported: 'Ambisonics', 'Binaural', 'Mono', 'CustomArrayIR'
    sample_rate = 16000
    for mode in ['example']:
        data_dir = f'alimeeting-data/{mode}-data'
        with open(f"data/{mode}_scene.txt", "r") as f:
            scene_list = f.readlines()
        scene_list = [scene.strip() for scene in scene_list]

        with open(f'{data_dir}/wav.scp', 'r')as f:
            reco2dir = f.readlines()
        reco2paths = [{reco.split('\t')[0]: reco.strip().split('\t')[1]} for reco in reco2dir]
        reco2dur = tool_utils.load_reco2dur(f'{data_dir}/reco2dur')
        reco2num_spk = tool_utils.load_reco2num_spk(f'{data_dir}/reco2num_spk')
        rttm = tool_utils.load_rttm(f'{data_dir}/rttm')
        with open(f"data/{mode}_speech.txt", "r") as f:
            speech_list = f.readlines()
        speech_list = [speech.strip() for speech in speech_list]

        noise_path = "data/noise.json"
        music_path = 'data/music.json'
        results_dir = f'audios/{mode}'
        logging.info(f"Simulating on {len(reco2paths)} audios with {len(scene_list)} scenes")

        if os.path.exists(results_dir):
            existed_reco = tool_utils.find_dirs_with_json(results_dir)
            remove_mic_recos = []
            # According to the audio generation order, once the last type of microphone array is generated, 
            # this reco can be skipped
            for r in existed_reco:
                if r.endswith('8circular'):
                    remove_mic_recos.append('_'.join(r.split('_')[:-1]))
            reco2paths = [reco2path for reco2path in reco2paths if list(reco2path.keys())[0] not in remove_mic_recos]
            exist_spks, exist_scenes = remove_spks_and_scenes(results_dir, existed_reco)
        

        for idx, reco2path in enumerate(reco2paths):
            start_time = time.time()

            total_time = 0.0
            reco = list(reco2path.keys())[0]
            path = list(reco2path.values())[0]
            # filter the rttm for reco
            reco_rttm = [line for line in rttm if line['reco'] == reco]
            # get the length of the reco
            length = reco2dur[reco]
            
            # Randomly select num_spk individuals from the speaker list and 
            # remove those who have been selected to avoid duplicate selections
            # as well as scene
            source_spks = random.sample(speech_list, int(reco2num_spk[reco]))
            speech_list = [speech for speech in speech_list if speech not in source_spks]
            scene = random.choice(scene_list)
            scene_list = [sc for sc in scene_list if sc != scene]
            
            os.makedirs(results_dir, exist_ok=True)
            logging.info(f"Processing {mode}-{reco} {idx+1}/{len(reco2paths)} on secene {scene}")
            
            generate_all_channels_for_a_wav(scene, sample_rate, results_dir, noise_path, music_path, source_spks, reco_rttm, reco, length)
            
            end_time = time.time()
            logging.info(f"Time elapsed: {(end_time - start_time)/60} min, Length of speech list: {len(speech_list)}")
            total_time += (end_time - start_time)/60
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Total time: {} min".format(total_time))
                
              
                
            