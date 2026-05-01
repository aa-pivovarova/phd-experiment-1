# phd-experiment-1

Currently working on fine-tuning a w2v2-bert-2.0 on the TORGO DB.
- functions.py - for all functions
- main.py - launch with `python main.py` and wait.

DOCKER INFORMATION
- Simply run docker
- Environment variables:
- - HF_TOKEN - huggingface personal write license token for downloads and checkpoints

<details>
<summary><b>Previous exp:</b></summary>
Currently working on fine-tuning a w2v2-bert-2.0 on the RU part of common voice 17.
- DataCollatorCTCWithPadding.py - a class file
- functions.py - for all functions
- main.py - launch with `python main.py` and wait.

DOCKER INFORMATION
- Using this image: `jupyter/tensorflow-notebook`
- Initialize a volume
- Port: 8888
- Environment variables:
- - HF_TOKEN - huggingface personal write license token for downloads and checkpoints
</details>