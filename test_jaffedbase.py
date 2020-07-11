import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
# from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import os
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import time
# USE_WEBCAM = True # If false, loads video file source
detector = MTCNN(min_face_size=10)
# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 25
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3] # why ?????
print(emotion_target_size)
# starting lists for calculating modes
emotion_window = []

# starting video streaming

#cv2.namedWindow('window_frame')
#video_capture = cv2.VideoCapture(0)

'''
# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source
'''
def get_namefile_detail():
    osname = '' #os.getcwd()
    list_emotion = os.listdir('/Volumes/NO NAME/emotions/jaffedbase')#'EmotionSet')#'/Volumes/NO NAME/FERG_DB_256/aia')
    if '._.DS_Store' in list_emotion:
        list_emotion.remove('._.DS_Store')
    '''
    osname+='/Volumes/NO NAME/emotions/jaffedbase'#'EmotionSet'#'/Volumes/NO NAME/FERG_DB_256/aia'
    osname_const = osname
    count = 0 
    list_emotion_images_namefile = []
    for emotion in  list_emotion:
        osname += '/' +emotion
        
        list_images = os.listdir(osname)
        
        if '.DS_Store' in list_images:
            list_images.remove('.DS_Store')
        count+=len(list_images)

        # for f in list_images :
        #     list_emotion_images_namefile.append(osname+'/'+f)
        # osname = osname_const
    '''
    #result_real = list_emotion_images_namefile[0].split('/')[1]
    print(list_emotion)
    return list_emotion
    #return list_emotion_images_namefile
happy = 0
neutral = 0
sad = 0
temp = 0
time_ = [] 
class TestScale:
    global time_
    global neutral,happy,sad
    global temp
    def __init__(self,namefile):
        self.namefile = namefile
    def emotion_detection(self):
        global time_
        global temp
        frame = cv2.imread(self.namefile)
        #print(frame)
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #MTCNN
        
        faces = detector.detect_faces(frame)
        faces = [face['box'] for face in faces]
        faces = faces
        print([faces])
        
        # exit()
        
        ''' HAARCASCADE 
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1,minNeighbors=1, minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)
        #print(faces)
        '''
        for face_coordinates in faces: #if 1 :
            
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            cv2.imshow('f',gray_face)
            # print(gray_face)
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size)) #interpolation : Linear
            except:
                #pass
                continue
            # print(gray_face)
            # print(gray_face.shape)
            b=time.time()
            gray_face = preprocess_input(gray_face, True)
            # print(gray_face)
            gray_face = np.expand_dims(gray_face, 0) #axis = 0
            # print(gray_face.shape)
            gray_face= np.expand_dims(gray_face,-1) #axis = -1
            # print(gray_face.shape)
            emotion_prediction = emotion_classifier.predict(gray_face)
            #tolist
            emotion_prediction = emotion_prediction.tolist()
            # print(emotion_prediction[0])
            emotion_prediction[0][0:3]=[0,0,0] #7
            emotion_prediction[0][5]=0 #7
            # print(emotion_prediction.shape)
            emotion_prediction = np.asarray(emotion_prediction)
            # print(emotion_prediction)
            emotion_probability = np.max(emotion_prediction)
            # print(emotion_probability)
            emotion_label_arg = np.argmax(emotion_prediction)
            # print(emotion_label_arg) # emotion numbers dict
            emotion_text = emotion_labels[emotion_label_arg]
            time_.append(time.time()-b)
            emotion_window.append(emotion_text)
            #print(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                pass
            if emotion_text:
                temp+=1
                return emotion_text
            else:
                pass
            # return emotion_mode
        
    def result_real(self):
        global neutral,happy,sad
        str_ = self.namefile
        val_0 = str_.split('/')[5]
        val = val_0.split('.')[1]
        #print(val)
        if "NE" in val :
            neutral +=1
            return "neutral"
        elif "HA" in val:
            happy +=1
            return "happy"
        elif "SA" in val:
            sad+=1
            return "sad"
        else:
            return val
        #return str_.split('.')[1][:3]
        #return str_.split('/')[5]
 #get_namefile_detail()
#for i in list_name:
    #print(i)
a = 0

'''
happy = len(os.listdir('/Volumes/NO NAME/emotions/jaffedbase'))#'EmotionSet/happy'))#'/Volumes/NO NAME/FERG_DB_256/aia/happy'))
neutral = len(os.listdir('/Volumes/NO NAME/FERG_DB_256/aia/surprise')) #'EmotionSet/neutral'))#'/Volumes/NO NAME/FERG_DB_256/aia/neutral'))
# surprise = 40
sad = len(os.listdir('/Volumes/NO NAME/FERG_DB_256/aia/sad'))#'EmotionSet/sad'))#'/Volumes/NO NAME/FERG_DB_256/aia/sad'))
'''
true_happy = []
true_neutral = []
# true_surprise = []
true_sad = []
#time_ = []

link = "/Volumes/NO NAME/emotions/jaffedbase/"
list_image = os.listdir(link)
list_image.remove("._.DS_Store")
list_image.remove(".DS_Store")
print(len(list_image))
for i in list_image:
    if "._" in i:
        continue
        i=i[2:]
    link+=i
    #print(link)
    try:
        
        TEST = TestScale(str(link))
        
        # print(TEST.result_real())
        
        #TEST.emotion_detection()
        
        if TEST.emotion_detection() == TEST.result_real():
            #time_.append((time.time()-b))
            if TEST.result_real() == "happy":
                true_happy.append(1)
            elif TEST.result_real() == "sad":
                true_sad.append(1)
            elif TEST.result_real() == "neutral":
                true_neutral.append(1)
            # elif TEST.result_real() == "surprise":
            #     true_surprise.append(1)

        a+=1
    except Exception as e:
        print(e)
        pass
    #fr = cv2.imread(link)
    #cv2.imshow('fr'+i,fr)
    link = "/Volumes/NO NAME/emotions/jaffedbase/"
'''
for i in list_name :
    #print(i)
    
    try:
        
        TEST = TestScale(str(i))
        
        # print(TEST.result_real())
        b=time.time()
        TEST.emotion_detection()
        time_.append((time.time()-b))
        if TEST.emotion_detection() == TEST.result_real():
            if TEST.result_real() == "happy":
                true_happy.append(1)
            elif TEST.result_real() == "sad":
                true_sad.append(1)
            elif TEST.result_real() == "neutral":
                true_neutral.append(1)
            # elif TEST.result_real() == "surprise":
            #     true_surprise.append(1)

        a+=1
    except Exception as e:
        print(e)
        pass
'''
neutral = 30
happy = 31
sad = 31
print(len(true_neutral),neutral)
print(a,temp)
print('Happy : scale = ',len(true_happy)/happy)
print('sad : scale = ',len(true_sad)/sad)
# print('surprise : scale = ',len(true_surprise)/surprise)
print('neutral : scale = ',len(true_neutral)/neutral)

divisions = ['happy','neutral','sad']
division_average_marks = [len(true_happy)/happy*100,len(true_neutral)/neutral*100,len(true_sad)/sad*100]
fig, ax = plt.subplots()
rects1 = ax.bar(divisions,division_average_marks,color = 'grey')
plt.title("Bar Emotion scale")
plt.ylim(0,100)
plt.xlabel("Emotions")
plt.ylabel("Mark : (%)")
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    i = 0
    global true_neutral,true_sad,true_happy
    for rect in rects:
        

        height = rect.get_height()
        if i==0:
            str_legend = str(len(true_happy))+'/31 = '+str(height)[:5]
        elif i == 1:
            str_legend = str(len(true_neutral))+'/30 = ' + str(height)[:5]
        else:
            str_legend = str(len(true_sad))+'/31 = ' + str(height)[:5]
        ax.text(rect.get_x() + rect.get_width()/2., 1.015*height,
                '%s' % str_legend,
                ha='center', va='bottom')
        i+=1
avg_time = 0
for i in time_:
    avg_time+=i

print(max(time_),min(time_),avg_time/len(time_))
autolabel(rects1)

plt.show()