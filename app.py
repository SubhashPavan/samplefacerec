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

app = Flask(__name__)

def detect_faces(img):

    img = Image.open(BytesIO(img))
    
    rects, landmarks = face_detect.detect_face(np.array(img),80);#min face size is set to 80x80
    aligns = []
    positions = []
    face_boundries = []
    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(160,np.array(img),landmarks[:,i])
        if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
            aligns.append(aligned_face)
            positions.append(face_pos)
        else: 
            print("Align face failed") #log        
    if(len(aligns) > 0):
        features_arr = extract_feature.get_features(aligns)
        no_of_faces = len(aligns)
        face_boundries = []
        for (i,rect) in enumerate(rects):
            rect = rect.tolist()
            face_boundries.append([rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]])

    else:
        no_of_faces = 0
    

    print (no_of_faces)
    
    return face_boundries
    
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
        face_boundries = detect_faces(image)
        return jsonify(faces=face_boundries)


if __name__ == "__main__":
    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    app.run(host="0.0.0.0", port=5555)
