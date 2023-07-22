import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")# set the alaram 

#ratio distance of different points in eyes
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    ear = (A+B)/(2.0*C)
    return ear

threshvalue = 0.25#threshold value expected when eyes are closed
least =60

#axis of eyes
(ls,le) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rs,re) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

flag =0

while True:
    set,vdframe = cap.read()
    vdframe = imutils.resize(vdframe,width=450)
    gray = cv2.cvtColor(vdframe,cv2.COLOR_BGR2GRAY ) #screen color into gray to get detect correctly
    subjects = detect(gray,0)

    for subject in subjects:
       shape = predict(gray,subject)
       shape = face_utils.shape_to_np(shape)
       lefteye= shape[ls:le]
       righteye = shape[rs:re]
       leftear = eye_aspect_ratio(lefteye)
       rightear = eye_aspect_ratio(righteye)
       ear = (leftear+rightear) / 2.0#finding average ratio
       lefteyehull = cv2.convexHull(lefteye) 
       righteyehull = cv2.convexHull(righteye)
       cv2.drawContours(vdframe,[lefteyehull] ,-1 , (0,255,0) ,1)
       cv2.drawContours(vdframe,[righteyehull] ,-1 , (0,255,0) ,1)
       if ear<threshvalue:
          flag+=1
          print(flag)
          if flag >= least :
             cv2.putText(vdframe,"ALERT!!!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
             mixer.music.play() #play the alaram
       else:
             flag = 0 


  
    cv2.imshow("Img",vdframe)
    key =  cv2.waitKey(1) & 0xFF
    if key == ord('q'): #enter q to exit
     break
cv2.destroyAllWindows()
cap.release()