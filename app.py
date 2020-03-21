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
from camera import VideoCamera

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
app = Flask(__name__)

def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):


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

def gen(camera):
    while True:
        frame = camera.get_frame()
        rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data = findPeople(features_arr,positions)
            for (i,rect) in enumerate(rects):
                print (rect)
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
                cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/recognize', methods=['GET'])
def recognize():
    if request.method == 'GET':
        camera_id = int(request.values.get('camid'))
        print (camera_id)
        return Response(gen(VideoCamera(camera_id)),mimetype='multipart/x-mixed-replace; boundary=frame') 
    

if __name__ == '__main__':
    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read())
    http_server = WSGIServer(('0.0.0.0',5000), app)
    http_server.serve_forever()        

