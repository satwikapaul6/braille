# import requirements needed
from flask import Flask, render_template
from utils import get_base_url

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index.html')

# define additional routes here
# route to handle the image processing and translation
@app.route(f'{base_url}/process_image', methods=['POST'])
def process_image():
    # get the uploaded image file
    image_file = request.files['image']
    
    # save the image file to a temporary location
    temp_image_path = '/path/to/temp/image.png'  # replace with the actual path
    
    image_file.save(temp_image_path)
    
    # process and translate the image
    translated_text = process_and_translate(temp_image_path)
    
    # render the result template and pass the translated text
    return render_template('result.html', translated_text=translated_text)

# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'url'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)


# process braille
from imutils.perspective import four_point_transform as FPT
from collections import Counter # if you pass in an iterable, counter counts each element and remembers how many times each element shows up
import matplotlib.pyplot as plt # executing plts
from imutils import contours # actually detects where there are concentrations of objects on our images, and identifies them as contours
from skimage import io # helps us plot some images
import numpy as np
import imutils
import cv2 # known as openCV
import re
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

import warnings
warnings.filterwarnings("ignore")


# hello