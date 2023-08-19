## copy required repo
# cd home
cd ../../../../
git clone https://github.com/microsoft/gpt-MT.git
git clone https://github.com/huggingface/peft.git
# install virtual env
conda create -n llmmt python=3.8
source /home/aiscuser/anaconda3/bin/activate llmmt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
cd LLMMT; git checkout 247673ff1226876f9635e7cbbfd5c35e3399a2f0; pip install -e ./; cd ..
cd peft; git checkout 029f416fced285fe72acd4e1be6c48067f7013d5; pip install -e ./; cd ..
pip install sentencepiece
pip install sacrebleu
pip install sacrebleu[ja]
pip install ipython
pip install pytest
pip install datasets
pip install evaluate
pip3 install deepspeed==0.9.3
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

## copy data
echo "Copying WMT data from /mnt/sdrgmainz01wus2/t-haoranxu/filtered_wmt22/.... It may take a while..."
cp -r /mnt/sdrgmainz01wus2/t-haoranxu/filtered_wmt22/ .
echo "Copying finished"
echo "Copying human-written data from  /mnt/sdrgmainz01wus2/t-haoranxu/wmt-flores200-dev-test/...."
cp -r /mnt/sdrgmainz01wus2/t-haoranxu/wmt-flores200-dev-test/ /home/aiscuser/
echo "Copying finished"

# cd workplace
cd ./LLMMT/examples/pytorch/llmmt
