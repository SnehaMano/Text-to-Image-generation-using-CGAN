import gradio as gr
import os
import io
import h5py
import torch
import ipywidgets
import numpy as np
from torch import nn
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer, XLNetModel
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

MODEL_PATH = "./checkpoint.pth"



device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=device)
models = checkpoint.get("models")

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = XLNetModel.from_pretrained("xlnet-base-cased")


    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden = self.transformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        context = hidden.mean(dim=1)
        context = context.view(*context.shape, 1, 1)
        return context
    

class Generator(nn.Module):
    def __init__(self, nz=100, nt=768, nc=3, ngf=64):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nt, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
        )

            # Completed - TODO: check out paper's code and add layers if required

            ##there are more conv2d layers involved here in 
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(ngf*8,ngf*2,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # nn.SELU(True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ngf*2,ngf*2,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # nn.SELU(True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ngf*2,ngf*8,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # nn.SELU(True),
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),   
            nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
        )
            
            # Completed - TODO: check out paper's code and add layers if required
            
            ##there are more conv2d layers involved here in 
            
        self.layer6 = nn.Sequential(
            nn.Conv2d(ngf*4,ngf,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # nn.SELU(True),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(ngf,ngf,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # nn.SELU(True),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(ngf,ngf*4,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # nn.SELU(True),
        )
        self.layer9 = nn.Sequential(  
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # nn.SELU(True),
            
            # state size. (ngf*2) x 16 x 16
        )
        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # nn.SELU(True),

            # state size. (ngf) x 32 x 32
        )
        self.layer11 = nn.Sequential(
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
  
    def forward(self,noise,encoded_text):
        x = torch.cat([noise,encoded_text],dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, nt=768):
        super().__init__()
        self.layer1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),

            nn.Conv2d(ndf*8,ndf*2,1,1),
            # nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer5 = nn.Sequential(

            nn.Conv2d(ndf*2,ndf*2,3,1,1),
            # nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer6 = nn.Sequential(

            nn.Conv2d(ndf*2,ndf*8,3,1,1),
            # nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.concat_image_n_text = nn.Sequential(
            nn.Conv2d(ndf * 8 + nt, ndf * 8, 1, 1, 0, bias=False), ## TODO: Might want to change the kernel size and stride
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 8, 2, 4, 1, 0, bias=False),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x, encoded_text):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = torch.cat([x, encoded_text.repeat(1, 1, 4, 4)], dim=1)
        x = self.concat_image_n_text(x)
        return x
    
te =  TextEncoder()
gen = Generator()
disc = Discriminator()

te.load_state_dict(models["text_encoder"])
te.eval()

gen.load_state_dict(models["generator"])
gen.eval()

disc.load_state_dict(models["discriminator"])
disc.eval()

models = {
    "text_encoder": te.to(device),
    "generator": gen.to(device),
    "discriminator": disc.to(device)
}

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
max_seq_len = 85

def prepareText(text):
        input_ids = tokenizer.encode(text)
        token_type_ids = [0] * (len(input_ids) - 1) + [1]
        attention_mask = [1] * len(token_type_ids)
        return {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask
        }

def padTokens(text_dict):
        pad_len = max_seq_len - sum(text_dict["attention_mask"])
        text_dict['input_ids'] =  [5] * pad_len + text_dict['input_ids']
        text_dict['token_type_ids'] =  [2] * pad_len + text_dict['token_type_ids']
        text_dict['attention_mask'] = [0] * pad_len + text_dict['attention_mask']   
        return text_dict

def preprocessText(text):
    return padTokens(prepareText(text))

def start(text):
    params = preprocessText(text)
    for key in params:
        params[key] = torch.tensor(params[key], device=device)[None]

    enc_text = models["text_encoder"](**params)
    noise = torch.randn((1, 100, 1, 1), device=device)
    gen_image = models["generator"](noise, enc_text).detach().squeeze().cpu()

    transform = transforms.ToPILImage()
    image = transform(gen_image)
    image.save("./image.jpg")
    return Image.open("./image.jpg")

face = gr.Interface(fn=start, inputs="text", outputs="image", title="Text to Image using Conditional Generative Adversarial Network", theme = gr.themes.Base(text_size = gr.themes.sizes.text_lg))
face.launch()
