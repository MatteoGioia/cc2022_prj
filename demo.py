#Dependencies
import streamlit as st
import matplotlib.pyplot as plt
import os

#Available datasets
models = ["apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps", "facades", "iphone2dslr_flower", "ae_photos"]

st.write("Demo start")

model = st.selectbox('Choose the model and dataset to use', models)
st.write("You selected {}".format(model))

download_ds_command = "./datasets/download_cyclegan_dataset.sh {}".format(model)
#Check dataset has not been downloaded already
os.system(download_ds_command)

download_md_command = "./scripts/download_cyclegan_model.sh {}".format(model)
#Check model has not been downloaded already
os.system(download_md_command)

model_pretrained = model + "_pretrained"
test_command = "python3 test.py --dataroot datasets/{}/testA --name {} --model test --no_dropout".format(model, model_pretrained)

img = plt.imread('./results/monet2photo_pretrained/test_latest/images/00550_fake.png')
plt.imshow(img)

img = plt.imread('./results/monet2photo_pretrained/test_latest/images/00550_real.png')
plt.imshow(img)