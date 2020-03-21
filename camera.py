import cv2

class VideoCamera(object):
    def __init__(self,camid):
        self.camid = camid
        self.video = cv2.VideoCapture(self.camid)
    
    def __del__(self):
        self.video.release()
         
    def get_frame(self):
        success, image = self.video.read()
        
        return image