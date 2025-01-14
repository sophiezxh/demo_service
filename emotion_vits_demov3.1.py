import os
import sys
sys.path.append(os.path.join("/", *os.getcwd().split(os.sep)))
from utils.text_utils import text_to_sequence
import utils.commons as commons
from utils.utils import get_hparams_from_file, load_checkpoint
from model.models import SynthesizerTrn
from utils.text.symbols import symbols
import IPython.display as ipd

import json
import tempfile
from moviepy.editor import VideoFileClip
import requests
import gradio as gr
from IPython.display import Audio
import soundfile as sf
from transformers import AutoProcessor, AutoModel, pipeline
import base64
import torch
import torchaudio
import funasr
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import unicodedata
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
os.environ['LANGUAGE'] = 'en'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import warnings
warnings.filterwarnings('ignore')
import logging
import urllib3
import botocore
import matplotlib
from PIL import Image

logging.getLogger('numba.core').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import PIL.Image
PIL.Image.DEBUG = False 
logging.getLogger("PIL").setLevel(logging.ERROR)

vl_model_dir = '/pathto/Qwen2-VL-7B-Instruct'
vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    vl_model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(vl_model_dir)


def video_understanding(video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": """Describe this video, tell me how many people are talking and detect the emotion of this video [neutral,sad,happy,angry,fear] . 
                                            You may ignore the subtitles. 
                                            You should pay most attention to the face expression of the people.
                                            """},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = vl_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f'video_content:{output_text}')
    return output_text


def clean_text(text):

    text = ''.join([char for char in text if char.isascii() or unicodedata.category(char).startswith('L')])

    text = re.sub(r'[\ud83d\ude0a?]', '', text)
    return text.strip()


def get_format_dialogue_data(video_content,query):
    import requests
    url = "http://qwen_llm_url"

    headers = {
        'Content-Type': 'application/json',
    }
    prompt = """Given a transcript of a dialogue or monologue, please analyze each segment's content and context step by step to reconstruct it into two-speaker dialogue or a monologue form.
    Example 1:
        Video content: "there are two people talking"
        Input: "Sorry for the interruption. Oh, that's fine. I was wondering, is there some place I could leave my bags. I'd like to go get something to eat. Certainly, sir, you can leave your bags here with me. And when your room is available, I'll have someone put it in your room. Thank you. Here is your registration card. Would you prefer a queen size or twin beds? Queen size, please. Okay, you will be in room 412. How would you like to pay for your room? With cash. Okay, Mr. Simmons, you'll check back with us in about an hour. Your room should be ready."
        Output: {"SPEAKER1": "Sorry for the interruption.", "SPEAKER2": "Oh, that's fine. I was wondering, is there some place I could leave my bags. I'd like to go get something to eat.", "SPEAKER1": "Certainly, sir, you can leave your bags here with me. And when your room is available, I'll have someone put it in your room.", "SPEAKER2": "Thank you.", "SPEAKER1": "Here is your registration card. Would you prefer a queen size or twin beds?", "SPEAKER2": "Queen size, please.", "SPEAKER1": "Okay, you will be in room 412. How would you like to pay for your room?", "SPEAKER2": "With cash.", "SPEAKER1": "Okay, Mr. Simmons, you'll check back with us in about an hour. Your room should be ready"}

    Example 2:
        Video content: "there is one person talking"
        Input: "I was wondering, is there some place I could leave my bags. I'd like to go get something to eat."
        Output: {"SPEAKER1": "I was wondering, is there some place I could leave my bags. I'd like to go get something to eat."}

    Note:
    1. Do not alter the original text content or change the order of words.
    2. Only return the "Output" part of the content; do not add anything else.
    Given:
    """

    query = f'''Video content: {video_content}
    Input: {query}
    '''

    data = {
        "promptTemplate": {
            "params": {
                "content": prompt + query
            },
            "templateId": "text_raw"
        }
    }
    print(f'===================get_format_dialogue_data debug propmt:{query}=========================')


    response = requests.post(url, json=data, headers=headers)
    print(f'===================get_format_dialogue_data debug replies.text:{response.text}=========================')
    try:
        # res = eval(response.text).get('response').get('choices')[0].get('message').get('content')
        res = eval(response.text.replace('null','999')).get('response').get('choices')[0].get('message').get('content')
    except:
        res = query
    res =  clean_text(res)
    return res


def get_zhijian_model_qa_data(video_content, dialogue_from_video, emotion):
    import requests
    url = "http://qwen_llm_url"
    
    headers = {
        'Content-Type': 'application/json',
    }

    prompt = f"""Given that the background of the conversation is: {video_content}, the content of the dialogue is: {dialogue_from_video}, and the reference emotion for the dialogue is: {emotion}.

    Please analyze the latest emotion in the current dialogue based on the above dialogue content, background description, and reference emotion [neutral, sad, happy, angry, fear].

    Then, continue the dialogue with a sentence of about 10-20 words, considering the context and emotion, to provide care and encouragement.

    The response_emotion should be chosen from the following options: [normal_angry, normal_calm, normal_fearful, normal_happy, normal_neutral, normal_sad, normal_surprised]

    Here are the basic rules for matching dialogue emotion (emotion) and response emotion (response_emotion) for your reference:
        1. emotion:happy corresponds to response_emotion:happy
        2. emotion:neutral corresponds to response_emotion: neutral or happy
        3. emotion:sad corresponds to response_emotion:neutral or calm or happy
        4. emotion:angry corresponds to response_emotion:angry or calm or neutral
        5. emotion:fear corresponds to response_emotion:fearful or calm or neutral
        6. The core purpose of choosing the response emotion is to provide care, encouragement, and comfort.

    ========================================
    For example:
        Given dialogue content: "SPEAKER1:Hey, girl, what time are we gonna go out tonight., SPEAKER2:I think I'm gonna stay home tonight., SPEAKER1:Come on. You said you're going out with me tonight. I've already changed. look, this is my dress., SPEAKER2:I feel that I really need some time to myself., SPEAKER1:I can't believe you...You are ditching me, SPEAKER2:I'm not ditching you, Come on.", the reference emotion is: sad
        Analysis: The conversation starts happily but ends up with one party feeling very sad and slightly angry because the date was canceled.
        Output: "emotion":"sad","response_emotion":"normal_happy","response":"this is the response sentence with sad emotion."
    ========================================
    The model should output the final result only, without any analysis or additional notes!

    Please output the content in the following format:
        {{ "emotion":"neutral", "response_emotion":"normal_neutral", "response":"this is the response sentence." }}
    """
    data = {
        "promptTemplate": {
            "params": {
                "content": prompt
            },
            "templateId": "text_raw"
        }
    }


    response = requests.post(url, json=data, headers=headers)

    return response.text


def extract_audio_from_video(video_path):

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip is not None:
            audio_clip.write_audiofile(temp_audio_file.name, codec='pcm_s16le')
            return temp_audio_file.name
        else:
            raise ValueError("Video does not contain an audio stream.")


def mock_reply_model(video_content, input_text, emotion_result):
    
    replies = get_zhijian_model_qa_data(video_content, input_text, emotion_result)

    print(f'1===========mock_reply_model debug replies:{replies}=============')

    try:
        res = eval(replies.replace('null','999')).get('response').get('choices')[0].get('message').get('content')
    except:
        res = replies
    print(f'===========mock_reply_model debug replies:{res}=============')

    pattern = r'\{[^}]*\}'
    res = re.findall(pattern, res, re.DOTALL)[0]

    return str(res)


def mock_emotion_model(llm_text):

    llm_text = llm_text.replace('```json','').replace('```','')
    print('=============mock_emotion_model debug llm_text============== \n',llm_text)
    try:
        emotions = eval(llm_text).get('emotion','This is a error response')
    except:
        emotions = llm_text.get('emotion','This is a error response')
    return emotions


def mock_response_emotion_model(llm_text):
     # { "emotion":"angry", "response_emotion":"normal_calm", "response":"I understand how you feel. Let’s find a way to move forward calmly and strategically." }
    llm_text = llm_text.replace('```json','').replace('```','')
    print('============= mock_response_emotion_model debug llm_text===============\n',llm_text)
    try:
        response_emotion = json.loads(llm_text).get('response_emotion','neutral')
    except:
        response_emotion = llm_text.get('response_emotion','neutral')
    return response_emotion


def mock_conversation_model(llm_text):
    # { "emotion":"angry", "response_emotion":"normal_calm", "response":"I understand how you feel. Let’s find a way to move forward calmly and strategically." }
    llm_text = llm_text.replace('```json','').replace('```','')

    print('============= mock_conversation_model debug llm_text===============\n',llm_text)
    try:
        responses = json.loads(llm_text).get('response','neutral')
    except:
        responses = llm_text.get('response','neutral')
        responses = responses.replace('SPEAKER1','').replace('SPEAKER2','')
    return responses


model_dir = '/pathto/SenseVoiceSmall'
vad_dir = '/pathto/speech_fsmn_vad_zh-cn-16k-common-pytorch'

asr_model = funasr.AutoModel(
    model=model_dir,
    # vad_model="fsmn-vad",
    vad_model=vad_dir,
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)


def mock_asr_model(audio_path):

    res = asr_model.generate(
        # input=f"{model_dir}/example/en.mp3",
        input=audio_path,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    return text


def text_to_speech1(text_input, response_emotion):
    '''
    chattts lib
    text_input： input for tts
    response_emotion: tts emotion(if need)
    '''
    import ChatTTS

    torchaudio.set_audio_backend("soundfile")

    chat = ChatTTS.Chat()
    # chat.load(source='custom',custom_path='./path_to_config_path.yaml',device='cuda') # Set to True for better performance
    # chat.load_models()
    chat.load()

    torch.manual_seed(2)
    rand_spk = chat.sample_random_speaker()

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # add sampled speaker 
        temperature = .3,   # using custom temperature
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
    )

    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
    )

    # text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
    print(f'============debug chatts text_input:{text_input}==========')
    text_input = clean_text(text_input)
    wavs = chat.infer(text_input, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
    file_name = "word_level_output.wav"
    torchaudio.save(file_name, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
    return file_name


def text_to_speech2(text_input, response_emotion):
    '''
    tencent cloud api
    text_input： input for tts
    response_emotion: tts emotion(if need)
    '''
    import hashlib
    import hmac
    import json
    import sys
    import time
    from datetime import datetime
    if sys.version_info[0] <= 2:
        from httplib import HTTPSConnection
    else:
        from http.client import HTTPSConnection


    def sign(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_id = "yourid"
    secret_key = "youkey"
    token = ""

    service = "tts"
    host = "tts.tencentcloudapi.com"
    region = ""
    version = "2019-08-23"
    action = "TextToVoice"

    endpoint = "https://tts.tencentcloudapi.com"
    algorithm = "TC3-HMAC-SHA256"
    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = "content-type:%s\nhost:%s\nx-tc-action:%s\n" % (ct, host, action.lower())
    signed_headers = "content-type;host;x-tc-action"
    emotion_dict = {}

    def result_match(response_emotion):

        emotion_dict = {"sad":'悲伤',"happy":'高兴',"angry":'生气',"fear":'恐惧',"netrual":'中性',"calm":'平静'}
        for emotion_ in emotion_dict.keys():
            if emotion_ in response_emotion:
                emotion_result_ = emotion_dict[emotion_]
                return emotion_result_
        return '中性'

    emotion_result_ = result_match(response_emotion)
    print(f'tencent API response_emotion：{response_emotion}，parameter emotion_result_：{emotion_result_}')

    import json
    data = {
        "Text": text_input,  
        "SessionId": "wdtyh",
        "EmotionCategory":emotion_result_,
        "VoiceType":601002,
    }


    payload = json.dumps(data, ensure_ascii=False)
    print(payload)
    # print(f'===========payload debug ==============:{payload}')

    params = json.loads(payload)
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                         canonical_uri + "\n" +
                         canonical_querystring + "\n" +
                         canonical_headers + "\n" +
                         signed_headers + "\n" +
                         hashed_request_payload)


    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = (algorithm + "\n" +
                      str(timestamp) + "\n" +
                      credential_scope + "\n" +
                      hashed_canonical_request)


    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": timestamp,
        "X-TC-Version": version
    }
    if region:
        headers["X-TC-Region"] = region
    if token:
        headers["X-TC-Token"] = token

    try:
        req = HTTPSConnection(host)
        req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
        response_json = req.getresponse()
        res = response_json.read()
        # print(f'--------------- RequestId debug:{json.loads(res).get("Response").get("RequestId")} -------------',)
        # print(res)
    except Exception as err:
        print(err)
    try:
        response = json.loads(res)
        base64_encoded_audio = response['Response']['Audio']
    except:
        base64_encoded_audio = response_json['Response']['Audio']

    audio_binary_data = base64.b64decode(base64_encoded_audio)
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_binary_data)
        temp_filename = temp_file.name

    return temp_filename



def text_to_speech3(input_text, response_emotion):
    '''
    doubao api
    text_input： input for tts
    response_emotion: tts emotion(if need)
    '''
    import json
    import uuid
    import requests


    appid = "9183243098"
    access_token= "yourtoken"
    cluster =  'volcano_tts'
    voice_type = "BV503_streaming"
    host = "openspeech.bytedance.com"
    api_url = f"https://{host}/api/v1/tts"
    header = {"Authorization": f"Bearer;{access_token}"}

    request_json = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": cluster
        },
        "user": {
            "uid": "yourid"
        },
        "audio": {
            "voice_type": voice_type,
            "encoding": "mp3",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text":input_text ,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson"

        }
    }

    resp = requests.post(api_url, json.dumps(request_json), headers=header)
    base64_encoded_audio = resp.json()["data"]

    audio_binary_data = base64.b64decode(base64_encoded_audio)
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_binary_data)
    return temp_file.name


def text_to_speech4(input_text, response_emotion):
    '''
    text_input： input for tts
    response_emotion: tts emotion(if need)
    '''
    # PROJECT_PATH = os.path.join('/', *os.getcwd().split(os.sep)[:-2])
    PROJECT_PATH = '/pathtodemo'
    hps_file = f"{PROJECT_PATH}/config/esd_en_e5.json"
    hps = get_hparams_from_file(hps_file)
    
    checkpoint_path = f"{PROJECT_PATH}/VITS_variant.pth"
    print(f'checkpoint_path:{checkpoint_path}')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device).eval()
    load_checkpoint(checkpoint_path, model, None)
    
    sid = 4
    text = input_text

    try:
        emotion_f_path = f"/pathto/mmefeature/{response_emotion}.pt"
    except:
        emotion_f_path = f"/pathto/mmefeature/happy.pt"
    # [normal_angry,normal_calm,normal_fearful,normal_happy,normal_neutral,normal_sad,normal_surprised]

    print(f'emotion_f_path:{emotion_f_path}')
    sid = torch.LongTensor([sid]).to(device)
    text = get_text(text, hps).unsqueeze(0).to(device)
    text_length = torch.LongTensor([text.size(1)]).to(device)
    emotion_f = torch.load(emotion_f_path).float().unsqueeze(0).to(device)
    with torch.no_grad():
        audio = model.infer(text, text_length, sid, emotion_f, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float()
    file_name = "emotion_vits_output.wav"
    torchaudio.save(file_name, audio.unsqueeze(0), hps.data.sampling_rate)
    return file_name



def emotion_llama_model(video_path, user_question):
    print(f'======emotion_llama_model video_input:{video_path}, user_question:{user_question}======')
    emotion_result = 'default'
    import requests

    url = "http://localhost:9999/emotion"

    params = {
        'video_path': video_path,
        'user_question': user_question
    }

    response = requests.get(url, params=params)
    print("API Response:", response)
    print("Response content:", response.text)

    if response.status_code == 200:
        emotion_result = response.text
        print("llama API emotion result:", emotion_result)
        return emotion_result

    return emotion_result


def video_to_text_and_reply(video_input, tts_option):
    print(f'======video_input:{video_input}, tts_option:{tts_option}======')
    if isinstance(video_input, dict): 
        video_path = video_input['path']
    elif isinstance(video_input, str):
        video_path = video_input
    else:
        raise TypeError(f"Unsupported input type: {type(video_input)}")

    video_content = video_understanding(video_path)
    print(f'========video_content=============: \n {video_content}')

    # extract audio
    audio_path = extract_audio_from_video(video_path)

    # ASR 2 text
    text_from_video = mock_asr_model(audio_path)
    print(f'========text_from_video(ASR)=============: \n {text_from_video}')

    # text 2 dialogue
    dialogue_from_video = get_format_dialogue_data(video_content, text_from_video)
    print(f'========dialogue_from_video=============: \n {dialogue_from_video}')

    # use emotion-llama to dectect speaker emotion
    user_question = f"text_from_video:{text_from_video}. [emotion] Please determine which emotion label in the video represents: happy, sad, neutral, angry, fear, surprise."
    emotion_result = emotion_llama_model(video_path, user_question)
    print("Emotion_llama_model Predicted Emotion:", emotion_result)

    # use LLM to decide the reply emotion and words
    llm_text = mock_reply_model(video_content, dialogue_from_video, emotion_result)
    print(f'========llm_text=============: \n {llm_text}')
    # { "emotion":"angry", "response_emotion":"normal_calm", "response":"I understand how you feel. Let’s find a way to move forward calmly and strategically." }

    # use LLM to decide the reply emotion
    response_emotion = mock_response_emotion_model(llm_text)

    # use LLM to response
    conversation_result = mock_conversation_model(llm_text)

    if tts_option == "chatts":
        temp_filename = text_to_speech1(conversation_result, response_emotion)
    elif tts_option == "tencent":
        temp_filename = text_to_speech2(conversation_result, response_emotion)
    elif tts_option == "doubao":
        temp_filename = text_to_speech3(conversation_result, response_emotion)
    elif tts_option == "Ours":
        temp_filename = text_to_speech4(conversation_result, response_emotion)

    else:
        temp_filename = None 

    data = {
        "1. ASR Content":text_from_video,
        "2. Emotion Result":emotion_result,
        "3. Response Emotion":response_emotion,
        "4. LLM Response": conversation_result
        }

    formatted_output = "\n".join(f"{key}:{value}" for key, value in data.items())

    return formatted_output, temp_filename


def generate_audio_only(tts_option, custom_reply_input, custom_tts_option, formatted_output):
    print(f'===============generate_audio_only debug custom_reply_input:{custom_reply_input}===============')

    formatted_output = formatted_output.replace('：',':')
    print(f'===============generate_audio_only debug formatted_output:{formatted_output}===============')

    # extract conversatio text
    conversation_result = formatted_output.split("4. LLM Response:")[1]
    print(f'===============generate_audio_only debug LLM Response:{conversation_result}===============')

    # extract emotion
    response_emotion = formatted_output.split("3. Response Emotion:")[1]
    response_emotion = response_emotion.split("4. LLM Response:")[0]
    print(f'===============generate_audio_only debug response_emotion:{response_emotion}===============')

    if not conversation_result and not custom_reply_input:
        return None

    if custom_reply_input:
        conversation_text = custom_reply_input
    else:
        conversation_text = conversation_result

    if custom_tts_option:
        response_emotion = custom_tts_option
    else:
        response_emotion = response_emotion

        
    if tts_option == "chatts":
        temp_filename = text_to_speech1(conversation_text, response_emotion)
    elif tts_option == "tencent":
        temp_filename = text_to_speech2(conversation_text, response_emotion)
    elif tts_option == "doubao":
        temp_filename = text_to_speech3(conversation_text, response_emotion)
    elif tts_option == "Ours":
        temp_filename = text_to_speech4(conversation_text, response_emotion)
    else:
        temp_filename = None 
    return temp_filename



custom_css = """
.gr-box {
    padding: 20px;
}
.gr-interface-container .gr-column {
    flex: 1 1 auto;
}
.gr-examples {
    height: 1200px; /* set Examples height */
    overflow-y: auto; /* add */
}
"""


file_list = os.listdir('./examples')
video_list = [f'examples/{file}' for file in file_list if '.mp4' in file]
print(f'video_list:{video_list}')



def _launch_demo():
    def regenerate_response(video_path, tts_option):
        reply, audio_filepath = video_to_text_and_reply(video_path, tts_option)
        return reply, audio_filepath

    def clear_content():
        
        return None, "", None, "Ours" 

    with gr.Blocks(css=custom_css) as demo:

        gr.Markdown(
            """
            <center><font size=4>This is a demo of Emotionally Adaptive Text-to-Speech System for Personalized and Multimodal Communication with the Elderly. Feel free to test and explore !!!🤗</font></center>
            """
        )
        gr.Markdown(
            """
            <center><font size=4>Upload a video, and I will perform the following steps:
            <br>1.ASR Content. 
            <br>2. Emotion Result. 
            <br>3. Response Emotion. 
            <br>4. Response Content.
            
            Finally, I will generate a voice response using an Emotional-adaptive method and play it back.</font></center>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # video_input = gr.Video(type="filepath", label="Upload Video")
                video_input = gr.Video(type="filepath", label="Upload Video", placeholder="Drag and drop a video file here or click to upload.")
                tts_option = gr.Dropdown(["chatts", "tencent", "doubao", "Ours"], value="Ours", label="TTS Options")
            
            with gr.Column(scale=2):
                # output_text = gr.Textbox(label="Generated Contents")
                output_text = gr.Textbox(label="Generated Contents", lines=8).style(container=True, css={"width": "80%", "height": "200px"})

                with gr.Row():
                    with gr.Column(min_width=300): 
                        custom_reply_input = gr.Textbox(label="Custom Reply Text")
                    with gr.Column(min_width=100):
                        custom_tts_option = gr.Dropdown(
                            ["normal_angry", "normal_calm", "normal_fearful", "normal_happy", "normal_neutral", "normal_sad", "normal_surprised"],
                            value="normal_neutral", label="Custom TTS Options")

                output_audio = gr.Audio(type="filepath", label="Generated Audio", lines=2)

        with gr.Row():
            generate_button = gr.Button("Generate Contents")
            regenerate_button = gr.Button("Regenerate Contents")
            clear_button = gr.Button("Clear Contents")
            audio_only_button = gr.Button("Generate Audio Only") 


        clear_button.click(clear_content, outputs=[video_input, output_text, output_audio, tts_option])


        generate_button.click(regenerate_response, inputs=[video_input, tts_option], outputs=[output_text, output_audio])


        regenerate_button.click(regenerate_response, inputs=[video_input, tts_option], outputs=[output_text, output_audio])


        audio_only_button.click(
            fn=generate_audio_only,
            inputs=[tts_option, custom_reply_input, custom_tts_option, output_text],
            outputs=output_audio
        )

        demo.title = "Web Demo Title"
        demo.description = "Upload a video, and I will generate a content summary based on the video's content, then play back a voice response."


        examples = gr.Examples(
            examples=video_list,
            inputs=[video_input],
            fn=lambda x: (None, "", "Ours"),
            outputs=[output_text, output_audio, custom_reply_input, tts_option],
            cache_examples=0
        )


    demo.launch(server_name='0.0.0.0',server_port=7860,share=True)


# launch Gradio application
if __name__ == "__main__":
    _launch_demo()

