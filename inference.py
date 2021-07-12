import argparse
import subprocess
import json
from utils.generic_utils import load_config

args = None
parser = None

INPUT_SENTENCE = "Text de test"

def set_parser():
    global parser
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',
                        type=str,
                        help='Text for which to generate speech')

def set_args():
    global args
    global INPUT_SENTENCE
    
    args = parser.parse_args()
    
    INPUT_SENTENCE = args.text

def set_configs():
    basePath = './'
    speakers_json = r'extern_tts_fork\model_tests\rss_speakers.json'
    name = 'rss'
    tts_pretrained_model = r'extern_tts_fork\checkpoints\ms_checkpoint_33550.pth.tar'

    CONFIG = load_config(r'extern_tts_fork\ro_config.json')
    CONFIG['run_description'] = 'Evaluating model trained on romanian RSS dataset from scratch.'
    CONFIG['run_name'] = 'rss_tacotron_run'
    CONFIG['datasets'] = [{
                    "name": name,
                    "path": "rss/",
                    "meta_file_train": "metadata_train.txt",
                    "meta_file_val": "metadata_val.txt"
                }]
    CONFIG['model_path'] = tts_pretrained_model
    CONFIG['speaker_id'] = 0
    CONFIG['speakers_json'] = speakers_json
    CONFIG['sentence'] = INPUT_SENTENCE
    with open('config.json', 'w') as fp:
        json.dump(CONFIG, fp)

    VC_CONFIG = load_config(r'extern_tts_fork\WaveRNN\config.json')
    VC_CONFIG['bits'] = 10
    VC_CONFIG['audio']['sample_rate'] = 22050
    VC_CONFIG["upsample_factors"] = [5, 5, 11]
    with open('wavernn_config.json', 'w') as fp:
        json.dump(VC_CONFIG, fp)
                
def run_inference():
    subprocess.run(["python",
                    r"extern_tts_fork\ro_synthesize.py",
                    r"config.json",
                    #".",
                    "--vocoder_path",
                    r"extern_tts_fork\checkpoints\true_rss_checkpoint_329000.pth.tar",
                    "--batched_vocoder",
                    "True",
                    "--vocoder_config_path",
                    r"wavernn_config.json",
                    "--use_cuda",
                    "True",
                    "--out_path",
                    "reply.wav"], stdout=subprocess.DEVNULL)

def main():
    set_parser()
    set_args()
    set_configs()
    run_inference()
    
if __name__ == "__main__":
    main()