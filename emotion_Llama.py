import os
import sys
sys.path.append(os.path.join("/", *os.getcwd().split(os.sep)))
from utils.text_utils import text_to_sequence
import utils.commons as commons
from utils.utils import get_hparams_from_file, load_checkpoint
from utils.text.symbols import symbols
import json
import tempfile
from moviepy.editor import VideoFileClip
import requests
import gradio as gr
import soundfile as sf
import base64
import torch
import torchaudio
import unicodedata
import re
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Supported CUDA architectures: {torch.cuda.get_arch_list()}")
import argparse
import os
import random
from collections import defaultdict

import cv2
import re
import numpy as np
from PIL import Image
import html
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from flask import Flask, request

app = Flask(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/demo.yaml',
                        help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args([
        '--cfg-path', 'eval_configs/demo.yaml',
        '--options', 'key1=value1', 'key2=value2'
    ])
    return args


args = parse_args()
print(args)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
print("----------args-----------:", args)
cfg = Config(args)


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        print("Error: Cannot read frame from video.")
        return None


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
llama_model = model_cls.from_config(model_config).to(device)
vis_processor_cfg = cfg.datasets_cfg.feature_face_caption.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
llama_model = llama_model.eval()


def emotion_llama_model(video_path, user_question):
    first_frame = get_first_frame(video_path)
    if first_frame is None:
        return "Error: Failed to get the first frame from the video."

    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    chat_state = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    chat = Chat(llama_model, vis_processor, device=device)
    img_list = []

    llm_message = chat.upload_img(video_path, chat_state, img_list)
    print(f'----------------gradio_answer llm_message:{llm_message}-----------')

    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)

    chat.ask(user_question, chat_state)

    print(f'----------------gradio_answer img_list:{img_list}-----------')
    print(f'----------------gradio_answer chat_state:{chat_state}-----------')
    llm_message = chat.answer(conv=chat_state, img_list=img_list, temperature=0.2, max_new_tokens=500, max_length=2000)[0]
    print(f'----------------gradio_answer llm_message:{llm_message}-----------')

    def extract_emotion(llm_message):
        import re

        emotion = 'neutral'
        emotion_pattern = r"(happy|sad|neutral|angry|fear|surprise)"
        match = re.search(emotion_pattern, llm_message)
        if match:
            emotion = match.group(1)

        return emotion

    llama_emotion = extract_emotion(llm_message.lower())
    return llama_emotion


@app.route('/emotion')
def hello_world():
    video_path = request.args.get('video_path')
    user_question = request.args.get('user_question')

    print(video_path)
    print(user_question)

    resp = emotion_llama_model(video_path, user_question)
    return resp


if __name__ == '__main__':
    app.run(port=9999,debug=True)