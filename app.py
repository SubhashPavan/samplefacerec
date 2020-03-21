import os
import sys
import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import json
import time
import numpy as np
from flask import Flask, redirect, url_for, request, Response, jsonify, redirect, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):

    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read())
    returnRes = []
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown"
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance
                    result = person
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes

def detect_faces(image):
    img = Image.open(BytesIO(image))
    rects, landmarks = face_detect.detect_face(np.array(img),80);#min face size is set to 80x80
    aligns = []
    positions = []

    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(160,np.array(img),landmarks[:,i])
        if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
            aligns.append(aligned_face)
            positions.append(face_pos)
        else: 
            print("Align face failed") #log        
    if(len(aligns) > 0):
        features_arr = extract_feature.get_features(aligns)
        recog_data = findPeople(features_arr,positions)
        people_names = []
        for x in recog_data:
            people_names.append(x[0])
        face_boundries = []
        for (i,rect) in enumerate(rects):
            rect = rect.tolist()
            face_boundries.append([rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]])
    try:
        return face_boundries, people_names
    except:
        return [], []

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    """
    curl -X POST -v -H "Content-Type: image/png" --data-binary @abba.png http://127.0.0.1:9099/prediction -o foo.jpg
    """
    if request.method == "POST":
        image = request.data
        face_coordinates, recog_data = detect_faces(image)
        
        return jsonify(faces=face_coordinates, people=recog_data)

if __name__ == '__main__':
    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    http_server = WSGIServer(('0.0.0.0',5000), app)
    http_server.serve_forever()        

