import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
#from train import gpu_checking
from torchvision import models
#from train import initial_classifier 
from PIL import Image
from gtts import gTTS
import os

def arg_parser():
    parser = argparse.ArgumentParser(description = 'classifier')
    parser.add_argument('--image', help = 'Point to image file for prediction', type = str, default='photo.png')
    parser.add_argument('--checkpoint', help = 'Point to checkpoint file as str' ,type = str, default='goods_checkpoint.pth')
    parser.add_argument('--top_k' , type = int , help = 'Choose top_k matches as int', default = '3')
    parser.add_argument('--category_names', dest = 'category_names', action = 'store', default ='goods_file.json')
    parser.add_argument('--gpu', dest = 'gpu' , action = 'store',default ='gpu',type = str)
    args = parser.parse_args()
    return args 

def checkpoint_loading (file_path):
    checkpoint = torch.load('checkpoint.pth', map_location='cpu')
    model = models.vgg16(pretrained = True)
    model.name = 'vgg16'
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def image_processing(image):
    with Image.open(image) as im:
        width = im.width
        height = im.height
        if width < height:
            width,height = 256,270
        else:
            width,height =270,256
        img = im.resize((width,height))
        
    center = width/4,height/4
    left,right,top,bottom =center[0]-(244/2),center[1]-(244/2),center[0]+(244/2),center[1]+(244/2)
        
    img = img.crop((left,right,top,bottom))
    rgb_img = img.convert('RGB')
    np_img = np.array(rgb_img)/255
        
    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]
    np_img = (np_img-mean)/std
    np_img = np_img.transpose(2,0,1)
        
    return np_img
    
def predict(model,top_k,cat_to_name,image):
    model.eval();
    model.to("cpu")
    tor_img = torch.from_numpy(np.expand_dims(image_processing(image),axis = 0)).type(torch.FloatTensor).to("cpu")
    logps = model.forward(tor_img)
    linps = torch.exp(logps)
    
    top_probs,top_labels = linps.topk(top_k)
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    return top_probs,top_labels,top_flowers

def print_probability(probs,flowers):
    for i,j in enumerate (zip(probs,flowers)):
        print("Rank:{} ".format(i+1),
              "Goods: {}, likelihood: {} %".format(j[1], ceil(j[0]*100)))
        
def main():
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

        model = checkpoint_loading(args.checkpoint)

        image_tensor = image_processing(args.image)
        print(image_tensor)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        top_probs, top_labels, top_flowers = predict(model,args.top_k, cat_to_name,args.image)
        print_probability(top_probs,top_flowers)
        if top_probs[0] > 0.6:
            text = top_flowers[0]
            result = "The predicted result good.. is {}".format(text)
        else:
            result = "The system  can't recognize   the good............... please  input a new  image"
        language = 'en'
        output = gTTS(text=result,lang=language,slow = False)
        output = output.save("output.mp3")
        os.system("start output.mp3")
if __name__ == '__main__': main()
    
    
   
        