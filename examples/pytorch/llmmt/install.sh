## copy required repo
# cd home
cd ../../../../
git clone https://github.com/microsoft/gpt-MT.git
git clone https://github.com/huggingface/peft.git
echo "Copying data from /mnt/sdrgmainz01wus2/t-haoranxu/filtered_wmt22/...."
cp -r /mnt/sdrgmainz01wus2/t-haoranxu/filtered_wmt22/ .
echo "Copying finished"
# install virtual env
conda create -n llmmt python=3.8
source /home/aiscuser/anaconda3/bin/activate llmmt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
cd LLMMT; pip install -e ./; cd ..
cd peft; pip install -e ./; cd ..
pip install sentencepiece
pip install sacrebleu
pip install sacrebleu[ja]
pip install ipython
pip install pytest
pip install datasets
pip install evaluate
pip3 install deepspeed
pip install einops
pip install wandb
pip install zstandard
pip install accelerate==0.20.3
pip install jsonlines

# install eval env
conda create -n comet python=3.8
source /home/aiscuser/anaconda3/bin/activate comet
pip install unbabel-comet
pip install sacrebleu[ja]
conda deactivate
# cd workplace
cd ./LLMMT/examples/pytorch/llmmt
