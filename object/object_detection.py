#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:00:59 2017

@author: punyajoy
"""


def access_dict(ob,width,height):
    a=[]
    for box,string in six.iteritems(ob):
      ymin, xmin, ymax, xmax=box
      xmin=int(xmin*width)
      ymin=int(ymin*height)
      ymax=int(ymax*height)
      xmax=int(xmax*width)
      nam=''
      j=0
      while string[0][j]!=':':
        nam=nam+string[0][j]
        j=j+1
      a.append((xmin,xmax,ymin,ymax,nam))
    return a  
def detect_object(sess,image_np):
      
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      with tf.device("/gpu:0"):
         (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
         ob=vis_util.visualize_boxes_and_labels_on_image_array(
                 image_np,
                 np.squeeze(boxes),
                 np.squeeze(classes).astype(np.int32),
                 np.squeeze(scores),
                 category_index,
                 use_normalized_coordinates=True,
                 line_thickness=8)
         return ob,image_np

def listen():
     r = sr.Recognizer()
     m = sr.Microphone()
     flag=0;
     speech.say("A moment of silence, please...")
     with m as source: r.adjust_for_ambient_noise(source)
     print("Set minimum energy threshold to {}".format(r.energy_threshold))
     for i in range(0,10):
         speech.say("Say something!")
         with m as source: audio = r.listen(source)
         speech.say("Got it! Now to recognize it...")
         try:
             # recognize speech using Google Speech Recognition
             value = r.recognize_google(audio)
             flag=1
             # we need some special handling here to correctly print unicode characters to standard output
             if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                 print(u"You said {}".format(value).encode("utf-8"))
             else:  # this version of Python uses unicode for strings (Python 3+)
                 print("You said {}".format(value))
         except sr.UnknownValueError:
             speech.say("Oops! Didn't catch that")
             flag=0
         except sr.RequestError as e:
             flag=0
             print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
         if flag==1:
             break
     return value 


     

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

class MOSSE:
    def __init__(self, frame, rect):
        x1, x2, y1, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            return

        self.pos = x+dx, y+dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1
     

def trackmosse(bbox,frame,cap):
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # set up the ROI for tracking
    tracker = MOSSE(frame_gray, bbox)
    
    while(1):
       
      ret ,frame = cap.read()
      if ret == True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tracker.update(frame_gray)
            vis = frame.copy()
            tracker.draw_state(vis)
            cv2.imshow('tracker state', tracker.state_vis) 
            
            # apply meanshift to get the new location
            
            # Draw it on image
            cv2.imshow('frame', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
      else:
            break
     
      fps.update()
    
 
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     

#def track(bbox,frame,cap):
#    tracker = cv2.Tracker_create("MIL")
#    ok = tracker.init(frame, bbox)
#    while True:
#        # Read a new frame
#        ok, frame = cap.read()
#        if not ok:
#            break
#         
#        # Update tracker
#        ok, bbox = tracker.update(frame)
# 
#        # Draw bounding box
#        if ok:
#            p1 = (int(bbox[0]), int(bbox[2]))
#            p2 = (int(bbox[1]), int(bbox[3]))
#            cv2.rectangle(frame, p1, p2, (0,0,255))
# 
#        # Display result
#        cv2.imshow("Tracking", frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
    

def track2(bbox,frame,cap):
    c=bbox[2]
    w=abs(bbox[3]-bbox[2])
    r=bbox[1]
    h=abs(bbox[1]-bbox[0])
    track_window = (c,r,w,h)
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    cv2.imwrite('img2.png',roi)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))  
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[16],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 
    while(1):
       
        ret ,frame = cap.read()
    
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('img2',img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        else:
            break


speech.say('hello.... i am would be humanoid module... lets start')

print("[INFO] sampling THREADED frames from webcam...")
#vs = WebcamVideoStream(src=0).start()

cap = cv2.VideoCapture(1)
with detection_graph.as_default():
    image_count=0
    fps = FPS().start()
    speech.say('do you want to track or detect?')
    print('do you want to track or detect?')
#    tracking=listen()
#    while (tracking !='detect' and tracking !='track'):
#        tracking=listen()
#    
        
    tracking='detect'
    while (True):
                 
     # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
     ret,image_np = cap.read()
     image_count=image_count+1
     if image_count%3==0 and tracking =='detect':
          ob,image_np=detect_object(sess,image_np)
          a=access_dict(ob,image_np.shape[0],image_np.shape[1])
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          cv2.imshow('image',image_np)
          if cv2.waitKey(1) & 0xFF == ord('q'):
               break
           
     elif tracking=='track':
         print('What object do you want to track?')
         speech.say('What object do you want to track?')
         str1='y'
         while str1=='y':
             #value=listen()
             value='person'
             speech.say('Do u want me to listen again?')
             str1=input('Do u want me to listen again(y/n)')
             
         print('hold the object steady for 10 sec')
         
         str1='y'
         count=0
         while (True):
             count=count+1
             ret,image_np = cap.read()
             ob,image_np=detect_object(sess,image_np)
             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
             cv2.imshow('image',image_np)
             key=cv2.waitKey(0);  
             if key == ord('q'):
               break
         a=access_dict(ob,image_np.shape[1],image_np.shape[0])
         flag=0
         for i in range(0,len(a)):
             if a[i][4] == value:
                 bbox=(a[i][0],a[i][1],a[i][2],a[i][3])
                 flag=1     
                 break
         
         if flag==1:
             speech.say('lets track')
             trackmosse(bbox,image_np,cap)
         break
         
         
         
      
     fps.update()
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
     # do a bit of cleanup 
#sess.close()
cv2.destroyAllWindows()
cap.release()








