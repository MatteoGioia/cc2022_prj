#Dependencies
import streamlit as st
import matplotlib.pyplot as plt
import os
from random import randint

def download_dataset(model):
    download_ds_command = "./datasets/download_cyclegan_dataset.sh {}".format(model)
    #Check dataset has not been downloaded already
    condition = "[[ ! -d datasets/{} ]] &&".format(model)
    os.system(condition + download_ds_command)

def get_reverse_model(model_name):
    if model_name == "apple2orange":
        return "orange2apple"
    elif model_name == "summer2winter_yosemite":
        return "winter2summer_yosemite"
    elif model_name == "horse2zebra":
        return "zebra2horse"
    elif model_name == "monet2photo":
        return "style_monet"
    else:
        print("This model has no reverse!")

def download_model(model):
    download_a2b_command = "./scripts/download_cyclegan_model.sh {} && ".format(model)
    reverse = get_reverse_model(model)
    download_b2a_command = "./scripts/download_cyclegan_model.sh {}".format(reverse)
    #Check model has not been downloaded already
    condition = "[[ ! -f checkpoints/{}_pretrained/latest_net_G.pth ]] &&".format(model)
    os.system(condition + download_a2b_command + download_b2a_command)

def test(model, force=False):

    nr_test_A = len(os.listdir("datasets/{}/testA".format(model)))
    nr_test_B = len(os.listdir("datasets/{}/testB".format(model)))

    model_pretrained = model + "_pretrained"
    rev_model_pretrained = get_reverse_model(model) + "_pretrained"

    test_command = "python3 test.py --dataroot datasets/{}/{} --name {} --model test --direction {} --num_test {} --no_dropout --gpu_ids -1"        
    condition = "[[ ! \"$(ls -A results/{}_pretrained/test_latest/images)\" ]] &&".format(model)
    
    if not force:
        os.system(condition + test_command.format(model, "testA", model_pretrained,"AtoB",nr_test_A) + " && " + test_command.format(model, "testB", rev_model_pretrained,"BtoA",nr_test_B))
    else:
        os.system(test_command.format(model, "testA", model_pretrained,"AtoB",nr_test_A) + " && " + test_command.format(model, "testB", rev_model_pretrained,"BtoA",nr_test_B))

def test_single(model, usr_dir, direction):

    model_pretrained = model + "_pretrained"
    test_command = "python3 test.py --dataroot {} --name {} --model test --direction {} --results_dir {} --no_dropout --gpu_ids -1"  
    os.system(test_command.format(usr_dir, model_pretrained, direction, usr_dir))

def get_img_list(model):
    img_list = os.listdir("results/{}_pretrained/test_latest/images".format(model))
    img_list = [img[:-9] for img in img_list]
    img_list = list(set(img_list))
    return img_list

#Available datasets
models = ["apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo", "iphone2dslr_flower"]

st.title("CycleGAN online demo")
model = st.selectbox('Choose the model and dataset to use', models, index=2)

#Download dataset, model
download_dataset(model)
download_model(model)

#Execute tests
test(model)

#Visualize two samples
st.subheader("Visualize two examples")
col1, col2 = st.columns(2)

#Get img list
mode = st.selectbox("Select mode", ["A2B","B2A"])
if mode == "A2B":
    img_list = get_img_list(model)
    used_model = model
else:
    img_list = get_img_list(get_reverse_model(model))
    used_model = get_reverse_model(model)

sample = st.selectbox("Select 2 samples", img_list)
sample = sample

with col1:
    img = plt.imread('./results/{}_pretrained/test_latest/images/{}_real.png'.format(used_model, sample))
    st.image(img, caption="real")

with col2:
    img = plt.imread('./results/{}_pretrained/test_latest/images/{}_fake.png'.format(used_model, sample))
    st.image(img, caption="fake")

#Upload your own sample
file = st.file_uploader(label="Upload your own sample", type=["png", "jpg"],
        accept_multiple_files=False)
mod = st.selectbox("Select mode", ["A2B", "B2A"], key="sb2")

if mod == "A2B":
    chosen_model = model
else:
    chosen_model = get_reverse_model(model)

if file is not None:
    #Create directory for user files
    usr_dir = "usr_imgs/{}/{}".format(str(randint(1, 10000)) + chosen_model, file.name[:-4])
    #Handle already existing directory / file
    if not os.path.isdir(usr_dir):
        os.makedirs(usr_dir)
    else:
        usr_dir = usr_dir + str(randint(1, 10000))
        os.makedirs(usr_dir)

    with open(os.path.join(usr_dir, file.name), "wb") as f:
        f.write(file.getbuffer())

    test_single(chosen_model, usr_dir, direction = mod)

    usr_col1, usr_col2 = st.columns(2)
    with usr_col1:
        img = plt.imread('{}/{}_pretrained/test_latest/images/{}_real.png'.format(usr_dir, chosen_model, file.name[:-4])) 
        st.image(img, caption="real") 
        
    with usr_col2:
        img = plt.imread('{}/{}_pretrained/test_latest/images/{}_fake.png'.format(usr_dir, chosen_model, file.name[:-4])) 
        st.image(img, caption="fake")

st.text("Credit: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix")