#!/bin/bash

#Install git
yum update -y
yum install git -y

#Clone cycle gan official repo
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix code
chown -R ec2-user code
cd pytorch-CycleGAN-and-pix2pix

#Install requirements
python3 -m pip install -r requirements.txt
