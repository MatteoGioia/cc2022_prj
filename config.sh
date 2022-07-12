#!/bin/bash

#Install git
yum update -y
yum install git -y

#Clone cycle gan official repo
cd /home/ec2-user
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix code
chown -R ec2-user code
cd code
chmod a+x scripts/download_cyclegan_model.sh

#Install requirements
python3 -m pip install -r requirements.txt
python3 -m pip install streamlit matplotlib plotly_express

wget https://raw.githubusercontent.com/MatteoGioia/cc2022_prj/main/demo.py 
chown ec2-user demo.py

mkdir .streamlit
wget https://raw.githubusercontent.com/MatteoGioia/cc2022_prj/main/config.toml
mv config.toml .streamlit

#Run demo automatically
streamlit run demo.py