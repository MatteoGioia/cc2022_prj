#!/bin/bash

#Install git
yum update -y
yum install git -y

#Clone cycle gan official repo
cd /home/ec2-user
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix code
chown -R ec2-user code
cd code

#Install requirements
python3 -m pip install -r requirements.txt
