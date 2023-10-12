##### Create environment
```bash
module load mamba/latest # only for Sol
mamba create -n pixel2cancer python=3.9
```

##### Installation
```
# install requirements
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# install cellular automata
cd ./Tumor_Synthesis/cellular
python setup.py install 

# install evaluation package
cd ./surface-distance
pip install .
```
