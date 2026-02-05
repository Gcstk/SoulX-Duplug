# SoulX-Duplug

*Preview Version of SoulX-Duplug*

<div align="center">
    <h1>
    SoulX-Duplug
    </h1>
    <p>
    Official code for <br>
    <b><em>SoulX-Duplug: </em></b>
    </p>
</div>


##  Overview
SoulX-Duplug is a plug-and-play streaming semantic VAD model designed for full-duplex speech dialogue.



## 🔥 Demos


For more examples, see demo page.


## 🚀 News
- 


## Install

### Clone and Install
Here are instructions for installing on Linux.
- Clone the repo
```bash
git clone https://github.com/Soul-AILab/SoulX-Duplug.git
cd SoulX-Duplug
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html

- Create Conda env:
```bash
conda create -n soulx-duplug -y python=3.10
conda activate soulx-duplug
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```



### Model Download

Download via hf:
```bash
# If you are in mainland China, please first set the mirror:
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Soul-AILab/SoulX-Duplug-0.6B --local-dir pretrained_models
```

Download via python:
```python
from huggingface_hub import snapshot_download
snapshot_download("Soul-AILab/SoulX-Duplug-0.6B", local_dir="pretrained_models") 
```

Download via git clone:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B pretrained_models
```



### Basic Usage
We have wrapped the model as a server. You can simply start it with the following commands:
```bash
bash run.sh
```

For usage (see [example_client.py](https://github.com/Soul-AILab/SoulX-Duplug/blob/main/example_client.py) for reference), streamingly send your audio query (in chunks) to the server, and the server will return its prediction of the current dialogue state in a `dict`:

- Format:
    ```python
    {
        "type": "turn_state",
        "session_id": ,         # session_id
        "state": {
            "state": ,          # predicted state: "idle", "nonidle", "speak", or "blank"
            "text": ,           # (optional) asr result of user's turn
            "asr_segment": ,    # (optional) asr result of current chunk
            "asr_buffer": ,     # (optional) asr result of last 3.2s
        },
        "ts": time.time(),      # timestamp
    }
    ```

- **"idle"** indicates that the current audio chunk contains no semantic content (e.g., silence, noise, or backchannel).

- **"nonidle"** indicates that the current audio chunk contains semantic content. In this case, `"asr_segment"` returns the ASR result of the current chunk, and `"asr_buffer"` returns the ASR result of the accumulated audio over the past 3.2 seconds.

- **"speak"** indicates that up to the current chunk, the user is judged to have stopped speaking and the utterance is semantically complete, meaning the system can take the turn. In this case, `"asr_segment"` returns the ASR result of the current chunk, `"asr_buffer"` returns the ASR result of the accumulated audio over the past 3.2 seconds, and `"text"` returns the complete transcription of the user’s utterance for this turn. 

- **"blank"** indicates that the current unprocessed streaming input does not yet fill a full chunk; the server has cached the input and is waiting for the next query.



### Dialogue System

We implemented a demo full-duplex speech dialogue system based on SoulX-Duplug. You can refer to [dialogue_system](https://github.com/Soul-AILab/SoulX-Duplug/blob/main/dialogue_system/README.md).


## TODOs
- [ ] Publish the technical report.


## Citation

```bibtex
coming soon
```

## License



## Acknowledgment
Great thank is given to QwenLM, GLM-4-Voice, [chinese_text_normalization](https://github.com/speechio/chinese_text_normalization), Paraformer, Sensevoice, and SLAM-LLM for their open-source contribution.


## Contact Us
