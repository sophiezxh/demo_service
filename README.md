
## 🛠️ Setup


### Preparing the Base(emotion_tts) Environment

``` python3.10 needed
pip install -r emotion_tts_requirements.txt
```

### Preparing the emotion_llama Environment

``` python3.10 needed
virtualenv emotion_llama_env 
source emotion_llama_env/bin/activate
pip install -r llama_test1_requirements.txt
```


Specify the path to Llama-2 in the [model config file](minigpt4/configs/models/minigpt_v2.yaml#L14):

```yaml
# Set Llama-2-7b-chat-hf path
llama_model: "checkpoints/Llama-2-7b-chat-hf"
```

Specify the path to HuBERT-large in the [conversation file](minigpt4/conversation/conversation.py#L263):

```yaml
# Set HuBERT-large model path
model_file = "checkpoints/transformer/chinese-hubert-large"
```


Specify the path to Emotion-LLaMA in the [demo config file](eval_configs/demo.yaml#L10):

```yaml
# Set Emotion-LLaMA path
ckpt: "checkpoints/save_checkpoint/Emoation_LLaMA.pth"
```

Launching Demo Locally

1. launch emotion-llama service at emotion_llama_env

```
source emotion_llama_env/bin/activate
python emotion_Llama.py
```
2. lauch demo at base 

```
python emotion_vits_demov3.1.py

Running on local URL:  http://0.0.0.0:7860
```
