# import requirements needed
from flask import Flask, render_template, request, url_for, send_from_directory
from utils import get_base_url
import os

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
@app.route(base_url + '/index.html')
def return_home():
    return render_template('index.html')

# toggle between about and home
@app.route(base_url + '/about.html')
def about():
    return render_template('about.html')


# route that will process our translation results
@app.route(base_url + '/process_image', methods = ['POST'])
def process_image():
    try:
        # get the uploaded image file
        image_file = request.files['image']

        # create a temporary directory
        temp_dir = 'uploaded_images'
        os.makedirs(temp_dir, exist_ok = True)

        # generate a unique filename for the uploaded image
        filename = 'uploaded_image.png'
        temp_image_path = os.path.join(temp_dir, filename)
        
        # save the image file to the temporary directory
        image_file.save(temp_image_path)
        
        # process and translate the image
        translated_text = process_and_translate(temp_image_path)
    
        # render the result template and pass the translated text and image URL
        image_url = url_for('uploaded_image', filename=filename)
        return render_template('result.html', Braille_Translation = translated_text, image_url=image_url)
    except Exception as e:
        return render_template('error.html')

# Route to serve the uploaded image
@app.route('/uploaded_images/<filename>')
def uploaded_image(filename):
    return send_from_directory('uploaded_images', filename)

# Error handling function
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(Exception)
def handle_exception(e):
    error_message = f"An error occurred: {str(e)}"
    return render_template('error.html'), 500




# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page


if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'url'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)


# PROCESS BRAILLE HERE
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
import PIL

import warnings
warnings.filterwarnings("ignore")

def process_and_translate(image_url):
    def get_image(url, iter = 2, width = None):
        image = cv2.imread(url)
        if width:
            image = imutils.resize(image, width)
        ans = image.copy()
        accumEdged = np.zeros(image.shape[:2], dtype="uint8")
        # convert image to black and white
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # get edges
        edged = cv2.Canny(gray, 75, 200)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        # get contours
        ctrs = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ctrs = imutils.grab_contours(ctrs)
        docCnt = None

        # ensure that at least one contour was found
        if len(ctrs) > 0:
            # sort the contours according to their size in
            # descending order
            ctrs = sorted(ctrs, key=cv2.contourArea, reverse=True)

            # loop over the sorted contours
            for c in ctrs:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)

                # if our approximated contour has four points,
                # then we can assume we have found the paper
                if len(approx) == 4:
                    docCnt = approx
                    break

        paper = image.copy()

        # apply Otsu's thresholding method to binarize the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = np.ones((5,5), np.uint8)
        # erode and dilate to remove some of the unnecessary detail
        thresh = cv2.erode(thresh, kernel, iterations = iter)
        thresh = cv2.dilate(thresh, kernel, iterations = iter)

        # find contours in the thresholded image
        ctrs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctrs = imutils.grab_contours(ctrs)

        return image, ctrs, paper, gray, edged, thresh


    def sort_contours(ctrs):
        BB = [list(cv2.boundingRect(c)) for c in ctrs]
        # choose tolerance for x, y coordinates of the bounding boxes to be binned together
        tol = 0.7*diam

        # change x and y coordinates of bounding boxes to their corresponding bins
        def sort(i):
            S = sorted(BB, key = lambda x: x[i])
            s = [b[i] for b in S]
            m = s[0]

            for b in S:
                if m - tol < b[i] < m or m < b[i] < m + tol:
                    b[i] = m
                elif b[i] > m + diam:
                    for e in s[s.index(m):]:
                        if e > m + diam:
                            m = e
                            break
            return sorted(set(s))
        # lists of of x and y coordinates basically gives all the coordinates for where a contour is located
        xs = sort(0)
        ys = sort(1)

        (ctrs, BB) = zip(*sorted(zip(ctrs, BB), key = lambda b: b[1][1]*len(image) + b[1][0]))
        # return the list of sorted contours and bounding boxes
        return ctrs, BB, xs, ys

    def get_diameter():
        boundingBoxes = [list(cv2.boundingRect(c)) for c in ctrs]
        c = Counter([i[2] for i in boundingBoxes])
        mode = c.most_common(1)[0][0]
        if mode > 1:
            diam = mode
        else:
            diam = c.most_common(2)[1][0]
        return diam

    def get_max_diam():
        boundingBoxes = [list(cv2.boundingRect(c)) for c in ctrs]
        diam = max([i[2] for i in boundingBoxes])
        return diam

    def get_circles():
        questionCtrs = []
        for c in ctrs:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

        # in order to label the contour as a character, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if diam * 0.8 <= w <= max_diam * 1.2 and diam * 0.8 <= h <= max_diam * 1.2 and 0.8 <= aspect_ratio <= 1.2:
            questionCtrs.append(c)
        return questionCtrs

    def draw_contours(questionCtrs):
        color = (0, 255, 0)
        i = 0
        for q in range(len(questionCtrs)):
            cv2.drawContours(paper, questionCtrs[q], -1, color)
            i += 1

    def get_spacing():

        def spacing(x):
            space = []
            coor = [b[x] for b in boundingBoxes]
            for i in range(len(coor)-1):
                c = coor[i+1] - coor[i]
                if c > diam//2: 
                    space.append(c)
            return sorted(list(set(space)))

        sX = spacing(0)
        sY = spacing(1)

        # smallest x-serapation (between two adjacent dots in a letter)
        m = min(sX)
        c = 0
        d1 = sX[0]
        d2 = 0
        d3 = 0

        for x in sX:
            if d2 == 0 and x > d1*1.3:
                d2 = x # this is the second smallest spacing found
            if d2 > 0 and x > d2*1.3:
                d3 = x # this is the third smallest spacing found, aka the largest amongst d1 and d2
                break

        linesV = [] # a list of all the x coordinates of where the vertical lines will fall
        linesH = [] # a list of all the y coordinates of where the horizontal lines will fall
        spaceListY = [] # a list of spaces Horizontal
        spaceListX = [] # a list of spaces Vertical
        isSingleLine = False


        # HOW DO WE DECIDE TO PLOT OUR LINES HERE??? here is the part where we decide the coordinates of where the lines will go themselves
        # hint: use ys, the list of all the y coordinates where all the contours are found

        # HORIZONTAL LINES STUFF
        linesH.append(min(ys))

        current_lineY = 0
        previous_lineY = min(ys)

        for i in ys: # i itself is each element in ys
            current_lineY = i
            if (current_lineY - previous_lineY) >= diam:
                spaceListY.append(current_lineY - previous_lineY)
                linesH.append(i)
                previous_lineY = i

        # this is a way to ensure that the horizontal lines are spaced properly in between lines of braille
        space_marked = False
        index = 0
        if len(spaceListY) > 1:
            while index < len(spaceListY):
                if spaceListY[index] > spaceListY[1] * 2:
                    space_marked = True
                    linesH.insert(index + 1, spaceListY[index - 1] + linesH[index])
                    spaceListY.pop(index)
                    spaceListY.insert(index, linesH[index] - linesH[index - 1])
                    spaceListY.insert(index + 1, linesH[index + 2] - linesH[index + 1])
                    index += 2
                else:
                    index += 1

        linesH.append(max(ys) + spaceListY[-1]) # the final line at the end of the page


        # VERTICAL LINE STUFF
        linesV.append(min(xs))

        current_lineX = 0
        previous_lineX = min(xs)

        for i in xs: # i itself is each element in ys
            current_lineX = i
            if (current_lineX - previous_lineX) >= diam:
                spaceListX.append(current_lineX - previous_lineX)
                linesV.append(i)
                previous_lineX = i

        # Spacing Horizontal
        minSpaceY = spaceListY[0]

        for i in range(len(spaceListY)):
            if spaceListY[i] < minSpaceY:
                minSpaceY = spaceListY[i]

        holderY = round(1/8 * minSpaceY, 0)
        indexListY = []
        for i in range(len(spaceListY)):
            if spaceListY[i] > minSpaceY + holderY:
                indexListY.append(i)

        if len(spaceListY) <= 2:
            isSingleLine = True

        #Spacing Vertical
        minSpaceX = spaceListX[0]
        maxSpaceX = max(spaceListX)

        for i in range(len(spaceListX)):
            if spaceListX[i] < minSpaceX:
                minSpaceX = spaceListX[i]
        if isSingleLine and spaceListX[0] > minSpaceX + round(1/8 * minSpaceX, 0):
            linesV.insert(0,linesV[0]- minSpaceX)
            spaceListX.insert(0,minSpaceX)

        holderX = round(1/8 * minSpaceX, 0)
        indexListX = []

        i = 0
        if len(spaceListX) > 1:
            while i < len(spaceListX):
                if spaceListX[i] > spaceListX[1] + holderX + diam:
                    spaceListX[i] = minSpaceX
                    linesV.insert(i + 1,linesV[i]+ minSpaceX)
                    spaceListX.insert(i+1, linesV[i+2]- linesV[i + 1])
                i += 1
        return linesV, linesH, d1, d2, d3, sX, sY, spaceListX, spaceListY, isSingleLine, space_marked


    def display_contours_and_lines(space_marked, figsize = (15,30), lines = False):
        #fig = plt.figure(figsize = figsize)
        #plt.rcParams['axes.grid'] = False
        #plt.rcParams['axes.spines.left'] = False
        #plt.imshow(paper)
        linesV_plotted = []
        linesH_plotted = []

        if lines:
            if space_marked:
                y = 0
                while y < len(linesH):
                    if y % 4 == 0 or y % 4 == 3:
                        #plt.axhline(linesH[y], color = 'blue')
                        linesH_plotted.append(linesH[y])
                    y += 1
            else:
                for y in range(0, len(linesH), 3):
                    #plt.axhline(linesH[y], color = 'blue')
                    linesH_plotted.append(linesH[y])
            for x in range(0, len(linesV), 2):
                #plt.axvline(linesV[x], color = 'red')
                linesV_plotted.append(linesV[x])

        if len(linesV_plotted) > 1:
            last_line = min(1500, (linesV_plotted[-1] + (linesV_plotted[-1] - linesV_plotted[-2]))) # the final line at the right hand side of our image
        else:
            last_line = 1500
        #plt.axvline(last_line, color = 'red')
        linesV_plotted.append(last_line)

        #plt.show()
        return sorted(linesV_plotted), sorted(linesH_plotted)
        #PLOT LINES HERE where we physically take th4 coordinates and plot the lines

   
    image, ctrs, paper, gray, edged, thresh = get_image(image_url, iter = 0, width = 1500)
    diam = get_diameter()
    max_diam = get_max_diam()
    ctrs, boundingBoxes, xs, ys = sort_contours(ctrs)
    questionCtrs = get_circles()
    draw_contours(questionCtrs)
    linesV, linesH, d1, d2, d3, sX, sY, spaceListX, spaceListY, isSingleLine, space_marked = get_spacing()

    plottedxs, plottedys = display_contours_and_lines(space_marked, (25,30), True)

    cropped_images = []
    for y_start, y_end in zip(plottedys[:-1], plottedys[1:]):
        for x_start, x_end in zip(plottedxs[:-1], plottedxs[1:]):
            grid_rectangle = image[y_start:y_end, x_start:x_end]
            cropped_images.append(grid_rectangle)

    def is_a_space(image, threshold = 250):
        # Convert image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the average pixel value
        average_pixel_value = np.mean(grayscale_image)

        # Check if the average pixel value is above the threshold
        return average_pixel_value >= threshold


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and feature extractor
    model_name = "satwikapaul/braille_4"
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model = model.to(device)

    model.eval()

    def get_predictions():
        # Process each cropped image and obtain predictions
        predictions = ''
        last_space_checker = False
        nextLetterCapital = False
        for image in cropped_images:
            is_space = is_a_space(image)
            if is_space:
                if not last_space_checker:
                    predictions += ' '
                    last_space_checker = True
            else:
                last_space_checker = False
                # Convert the image to a PyTorch tensor and move it to the device
                image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).to(device)
                inputs = extractor(images=image_tensor, return_tensors="pt")

                # Move the input tensors to the device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass through the model
                with torch.no_grad():
                    outputs = model(**inputs)

                # Obtain the predicted class probabilities
                predicted_probs = torch.softmax(outputs.logits, dim=1)

                # Obtain the predicted class label
                predicted_label = torch.argmax(predicted_probs, dim=1).item()

                # Get the corresponding class name
                if predicted_probs[0, predicted_label] > .50:
                    predicted_class = model.config.id2label[predicted_label]
                    if predicted_class == 'period' or predicted_class == 'capital' or predicted_class == 'question%20mark':
                        if predicted_class == 'period':
                            predictions += '.'
                        elif predicted_class == 'capital':
                            nextLetterCapital = True
                        elif predicted_class == 'question%20mark':
                            predictions += '?'
                    else:
                        if nextLetterCapital:
                            predictions += predicted_class.upper()
                            nextLetterCapital = False
                        else:
                            predictions += predicted_class
        return predictions

    translated_text = get_predictions()
    return translated_text