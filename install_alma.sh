# Install training and generation env
pip install git+https://github.com/fe1ixxu/ALMA.git@hf-install
pip install peft==0.4.0
pip install sentencepiece
pip install sacrebleu
pip install sacrebleu[ja]
pip install ipython
pip install pytest
pip install datasets
pip install evaluate
pip3 install deepspeed==0.10.0
pip install einops
pip install wandb
pip install zstandard
pip install accelerate==0.21.0
pip install jsonlines

# # install eval env
# conda create -n comet python=3.8
# source /home/aiscuser/anaconda3/bin/activate comet
# pip install unbabel-comet
# pip install sacrebleu[ja]
