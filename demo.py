#Dependencies
import streamlit as st
import matplotlib.pyplot as plt
import os

def download_dataset(model):
    download_ds_command = "./datasets/download_cyclegan_dataset.sh {}".format(model)
    #Check dataset has not been downloaded already
    condition = "[[ ! -d datasets/{} ]] &&".format(model)
    os.system(condition + download_ds_command)

def download_model(model):
    download_md_command = "./scripts/download_cyclegan_model.sh {}".format(model)
    #Check model has not been downloaded already
    condition = "[[ ! -f checkpoints/{}_pretrained/latest_net_G.pth ]] &&".format(model)
    os.system(condition + download_md_command)

def test(model, force=False):

    nr_test_A = len(os.listdir("datasets/{}/testA".format(model)))
    nr_test_B = len(os.listdir("datasets/{}/testB".format(model)))

    model_pretrained = model + "_pretrained"
    test_command = "python3 test.py --dataroot datasets/{}/{} --name {} --model test --direction {} --num_test {} --no_dropout --gpu_ids -1"
    condition = "[[ ! \"$(ls -A results/{}_pretrained/test_latest/images)\" ]] &&".format(model)
    
    if not force:
        os.system(condition + test_command.format(model, "testA", model_pretrained,"AtoB",nr_test_A) + " && " + test_command.format(model, "testB", model_pretrained,"BtoA",nr_test_B))
    else:
        os.system(test_command.format(model, "testA", model_pretrained,"AtoB",nr_test_A) + " && " + test_command.format(model, "testB", model_pretrained,"BtoA",nr_test_B))

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
img_list = get_img_list(model)
sample = st.selectbox("Select 2 samples", img_list)
sample = sample

with col1:
    img = plt.imread('./results/{}_pretrained/test_latest/images/{}_fake.png'.format(model, sample))
    st.image(img, caption="fake")

with col2:
    img = plt.imread('./results/{}_pretrained/test_latest/images/{}_real.png'.format(model, sample))
    st.image(img, caption="real")

#Upload your own sample
file = st.file_uploader(label="Upload your own sample to conver from a A to B (warning: reloads website!", type=["png", "jpg"],
        accept_multiple_files=False)

test_dir = "./datasets/{}/testB".format(model)
if file is not None:
    with open(os.path.join(test_dir, "0.jpg"), "wb") as f:
        f.write(file.getbuffer())

    test(model, force=True)


st.text("Credit: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix")