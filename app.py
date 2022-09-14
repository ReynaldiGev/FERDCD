from cgitb import text
import base64
import time
from turtle import width
import requests
import csv
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from operator import truediv
from re import template
import tempfile
from textwrap import fill
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import glob as gb
import os
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import cv2
import moviepy.editor as moviepy
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

train_data_dir='dataset/train'
validation_data_dir='dataset/validation'
folder_path = 'D:/Facial-Emotion-Recognition-main/'

temp_file_to_save = './temp_file_1.mp4'
temp_file_result  = './temp_file_2.mp4'
temp_file_emot = './temp_file_3.csv'
temp_file_emot2 = './temp_file_4.txt'


st.set_page_config(
    page_title="DSS Project",
    page_icon="😉",
    layout="centered",
    initial_sidebar_state="auto")

hide_footer_style = """
        <style>
        footer {visibility: hidden;}
        </style
        """

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #D0050B;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #4F3E52;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)


st.markdown(hide_footer_style, unsafe_allow_html=True)



def get_counts(path):
  emotions = os.listdir(path)

  cls_counts = {}
  for emotion in emotions:
    count = len(os.listdir(os.path.join(path, emotion)))
    # print(emotion, count)
    cls_counts[emotion] = count

  return cls_counts
train_counts = get_counts(train_data_dir)

def eda_chart():
    fig = px.bar(
        x=train_counts.keys(), 
        y=train_counts.values(),
        title="layout.hovermode='y'")
    fig.update_traces(hovertemplate=None)
    fig.update_layout(barmode='relative',
                  title_text='Train classes Bar chart',
                  title_x=0.5,
                  xaxis_title="Emotion",
                  yaxis_title="Counts",
                  hovermode = "y")
    st.write("EDA Chart", fig)

def sample_angry():
    expression='angry'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_happy():
    expression='happy'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_neutral():
    expression='neutral'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_sad():
    expression='sad'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_surprise():
    expression='surprise'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_download = "https://assets1.lottiefiles.com/packages/lf20_KO31Fh.json"
lottie_download = load_lottieurl(lottie_url_download)

lottie_url_welcome = "https://assets9.lottiefiles.com/packages/lf20_7frdqxon.json"
lottie_welcome = load_lottieurl(lottie_url_welcome)

lottie_url_hello = "https://assets7.lottiefiles.com/private_files/lf30_6g1ve0mh.json"
lottie_hello = load_lottieurl(lottie_url_hello)

lottie_url_ty = "https://assets6.lottiefiles.com/packages/lf20_tjbhujef.json"
lottie_ty = load_lottieurl(lottie_url_ty)

def write_bytesio_to_file(filename2, bytesio):
            """
            Write the contents of the given BytesIO to a file.
            Creates the file or overwrites the file if it does
            not exist yet. 
            """
            with open(filename2, "wb") as outfile:
            # Copy the BytesIO stream to the output file
                outfile.write(bytesio.getbuffer())

face_classifier = cv2.CascadeClassifier(r'D:\Facial-Emotion-Recognition-main\haarcascade_frontalface_default.xml')
json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier =tf.keras.models.model_from_json(loaded_model_json)
classifier.load_weights('D:/Facial-Emotion-Recognition-main/model_weight.h5')
emotion_labels = ['Angry','Happy','Neutral', 'Sad', 'Surprise']



model=tf.keras.models.model_from_json(loaded_model_json)
model.load_weights('D:/Facial-Emotion-Recognition-main/model_weight.h5')
faceDetect=cv2.CascadeClassifier(r'D:\Facial-Emotion-Recognition-main\haarcascade_frontalface_default.xml')
labels_dict= {0:'Angry',1:'Happy',2:'Neutral',3:'Sad',4:'Surprise'}

def image_face_detected(image_in):
    frame=cv2.imread(image_in)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    get_label = []
    tempathasil = './fotohasil.jpg'
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        emot = labels_dict[label]
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        if emot !=0:
            get_label.append(emot)
        else:
            print('abc')
        cv2.imwrite(tempathasil, frame)

    buka = Image.open(tempathasil)
    #cv2.imshow("Frame",frame)
    st.image(buka)
    df = pd.DataFrame(get_label)
    df.columns = ["Expression"]
    df = df.value_counts().rename_axis('Expression').reset_index(name='counts')

    labels = pd.DataFrame(emotion_labels)
    labels.columns = ['Expression']

    df2 = pd.merge(labels, df, on='Expression', how='left')
    df2['counts'] = df2['counts'].fillna(0)

    df2.to_csv(temp_file_emot, index=False)



RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
                    def transform(self, frame):
                        img = frame.to_ndarray(format="bgr24")

                        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        faces = face_classifier.detectMultiScale(img_gray)
                        f=open(temp_file_emot2, 'a')

                        for (x,y,w,h) in faces:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) 
                            roi_gray = img_gray[y:y+h,x:x+w]
                            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
      
                            if np.sum([roi_gray])!=0:
                                roi = roi_gray.astype('float')/255.0  # normalizing
                                roi = tf.keras.preprocessing.image.img_to_array(roi)
                                roi = np.expand_dims(roi,axis=0)

                                prediction = classifier.predict(roi)[0]
                                label=emotion_labels[prediction.argmax()]
                                f.write(label+"\n")
                                label_position = (x,y-10)
                                cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                            else:
                                cv2.putText(img,'No Faces Found',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    

                            
                        return img


def statistics_visualization():
    df2 = pd.read_csv(temp_file_emot, encoding='utf-8')
    fig = go.Figure(data=go.Scatterpolar(r=df2['counts'],
      theta=df2['Expression'],
      fill='toself',
      hovertemplate = "<br>Emotion: %{theta} </br> Count: %{r} </br> "
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True
        ),
      ),
      showlegend=False
    )
    
    st.write("Emotion Radar",fig)

    fig2 = go.Figure(go.Bar(x=df2['counts'], 
                       y=df2['Expression'], 
                       orientation='h',
                       hovertemplate = "<br>Emotion: %{y} </br> Count: %{x} </br>" ))
    fig2.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    st.write("Emotion Barplot",fig2)

def statistics_visualizationtwo():
    txt =  pd.read_fwf('./temp_file_4.txt', header=None, names=['Expression'])
    df=pd.DataFrame(txt)
    df = df.value_counts().rename_axis('Expression').reset_index(name='counts')

    emotion_labels = ['Angry','Happy','Neutral', 'Sad', 'Surprise']
    labels = pd.DataFrame(emotion_labels)
    labels.columns = ['Expression']
    df2 = pd.merge(labels,df, on = 'Expression', how='left')
    df2['counts'] = df2['counts'].fillna(0)

    fig = go.Figure(data=go.Scatterpolar(r=df2['counts'],
      theta=df2['Expression'],
      fill='toself',
      hovertemplate = "<br>Emotion: %{theta} </br> Count: %{r} </br> "
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True
        ),
      ),
      showlegend=False
    )
    
    st.write("Emotion Radar",fig)

    fig2 = go.Figure(go.Bar(x=df2['counts'], 
                       y=df2['Expression'], 
                       orientation='h',
                       hovertemplate = "<br>Emotion: %{y} </br> Count: %{x} </br> " ))
    fig2.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    st.write("Emotion Barplot",fig2)

main_image = Image.open('D:/Facial-Emotion-Recognition-main/img/dl.png')
top_image = Image.open('D:/Facial-Emotion-Recognition-main/img/zz.png')
bot_image = Image.open('D:/Facial-Emotion-Recognition-main/img/aa.png')
def main():
    st.image(main_image,use_column_width='auto')
    st.title("Face Emotion Detection Application")
    with st.sidebar:
        st_lottie(lottie_welcome, key="welcome")
    #st.sidebar.image(top_image, use_column_width='auto')
    st.sidebar.header("Menu")
    with st.sidebar:
        choice = option_menu(
            menu_title="Select Activity",
            options=["Introduction", "Dataset", "Run App","About Me"],
            icons=["house", "folder2","cpu","person-lines-fill"],
            menu_icon="cast",
            default_index=0,
            styles={
                "nav-link-selected": {"background-color": "#2A9FB1"},
                "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#4F3E52"}
            }
        )

    if choice == "Introduction":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;font-size:18px;">
                                            This demo illustrates a combination of animation and plotting with
                                            Streamlit. It also provides three ways to detect the emotion of many people. Enjoy!🤗🥰</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["😉 FER", "🎰 CNN","🎯 Why Important?"])
        with tab1:
            st.header("Facial Emotion Recognition")
            st.image("https://recfaces.com/wp-content/uploads/2021/03/rf-emotion-recognition-rf-830x495-1.jpeg", width=600)
            st.write("""Facial Emotion Recognition (FER) is the technology that analyses facial expressions from
            both images and videos in order to reveal information on one’s emotional state. This technology used for analysing sentiments from non verbai communication.""")
        
        with tab2:
            st.header("Convolutional Neural Network")
            st.image("https://149695847.v2.pressablecdn.com/wp-content/uploads/2018/01/nural-network-05.jpg", width=600)
            file_ = open("D:/Facial-Emotion-Recognition-main/img/cnn.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="cnn gif" width="600">',
                unsafe_allow_html=True,
            )
        with tab3:
            st.header("Business impact ✅")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Salesment 📈")
                st.video("https://youtu.be/lDC90ObdMEs")

            with col2:
                st.subheader("Automotive 🚙")
                st.video("https://youtu.be/bKRd2Ux0QuE")
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Education 🏫")
                st.video("https://youtu.be/RbpUX4KD3iQ")

            with col2:
                st.subheader("Politic 🪙")
                st.video("https://youtu.be/94Hb-RdfQpo")


    elif choice == "Dataset":
        st.header("**About dataset🌍**")
        st.write("""
                 This data is taken from Kaggle under the title **"Facial Expression Recognition 2013 (FER 2013)"**. 
                 In this project, a facial expression data will be used which is divided into 5 types of expressions.
                 """)
        with st.expander("See Data Visualization"):
            eda = Image.open("D:/tes/FER/img/bardata.jpg")
            st.image(eda, width=300)

        st.markdown("""---""")

        st.header("**Sample Image🔍**")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Angry", "Happy","Neutral","Sad","Surprise"])
        with tab1:
            sample_angry()
        with tab2:
            sample_happy()
        with tab3:
            sample_neutral()
        with tab4:
            sample_sad()
        with tab5:
            sample_surprise()
        
    elif choice == "Run App":
        st.sidebar.subheader("How would you like to use?")
        with st.sidebar:
            mediapg = option_menu(
                menu_title="Pick channel",
                options=["Video","Image","Live Camera"],
                icons=["film", "images","webcam"],
                default_index=0,
                styles={
                    "nav-link-selected": {"background-color": "#2A9FB1"},
                    "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#4F3E52"}
                    }
            )
        
        if mediapg == "Video":
             uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov','webm'])
             def video():
                write_bytesio_to_file(temp_file_to_save, uploaded_video)
                cap = cv2.VideoCapture(temp_file_to_save)
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_fps = cap.get(cv2.CAP_PROP_FPS)  
                st.write(width, height, frame_fps)
                out = cv2.VideoWriter(temp_file_result, fourcc, frame_fps, (width, height))
                get_label = []
                while True:
                    ret, frame=cap.read()
                    if not ret: break
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces=face_classifier.detectMultiScale(gray, 1.3, 5)
                    
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) 
                        roi_gray = gray[y:y+h,x:x+w]
                        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
                        if np.sum([roi_gray])!=0:
                            roi = roi_gray.astype('float')/255.0  
                            roi = tf.keras.preprocessing.image.img_to_array(roi)
                            roi = np.expand_dims(roi,axis=0)

                            prediction = classifier.predict(roi)[0]
                            label=emotion_labels[prediction.argmax()]
                            label_position = (x,y-10)
                            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        else:
                            cv2.putText(frame,'No Faces Found',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        
                        if label !=0:
                            get_label.append(label)
                        else:
                            print('abc')
                    #cv2.imshow('Emotion Detector',frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    out.write(frame)
                    
                out.release() 
                cap.release()
                cv2.destroyAllWindows()

                output_video = open(temp_file_result,'rb')
                out_bytes = output_video.read()
                st.video(out_bytes)
                st.write("Detected Video")

                df = pd.DataFrame(get_label)
                df.columns = ["Expression"]
                df = df.value_counts().rename_axis('Expression').reset_index(name='counts')
                labels = pd.DataFrame(emotion_labels)
                labels.columns = ['Expression']

                df2 = pd.merge(labels, df, on='Expression', how='left')
                df2['counts'] = df2['counts'].fillna(0)
                df2.to_csv(temp_file_emot, index=False)

             if uploaded_video != None:
                vid = uploaded_video.name
                with open(vid, mode='wb') as f:
                    f.write(uploaded_video.read()) # save video to disk
                
                st_video = open(vid,'rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("Uploaded Video")

                st.markdown("""---""")

                st.text("Press Process to display the face emotion detected image.")
                if st.button('Process', key='pross'):
                    with st_lottie_spinner(lottie_download):
                        video()
                    with st.expander("See Result"):
                        statistics_visualization()


        elif mediapg == "Image":
            uploaded_image = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])
            if uploaded_image is not None:
                image1 = Image.open(uploaded_image)
                st.image(image1)
                st.text('Uploaded Image')
                #st.write(uploaded_image.name)

                st.markdown("""---""")

                st.text('Press Process to display the face emotion detected image')
                if st.button('Process'):
                    with st_lottie_spinner(lottie_download):
                        image_face_detected(uploaded_image.name)
                    with st.expander("See Result"):
                        statistics_visualization()

        elif mediapg =="Live Camera":
            with open(temp_file_emot2, 'a'):pass
            st.header("Live Stream")
            st.write("Click on start to use webcam and detect your face emotion")
            webrtc_streamer(key="example", video_processor_factory=Faceemotion,mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION)
            
            st.markdown("""---""")
            
            if st.button('Process'):
                with st.expander("See Result"):
                    statistics_visualizationtwo()


    elif choice == "About Me":
        st.markdown("<h1 style='text-align: center;</h1> [![Foo](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/reynaldihutapea/)", unsafe_allow_html=True)
        html_temp_home1 = """<div style="padding:10px">
                                        <h4 style="color:white;text-align:center;font-size:18px;">
                                        Hey this is Reynaldi Gevin 😄</h4>
                                        <h4 style="color:white;text-align:center;font-size:18px;">
                                        If you are interested in this project, we can connect by:</h4>
                                        </div>
                                        </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

  
        column1, column2, column3, column4, column5, column6, column7= st.columns(7)
        with column1:
            st.write("")
        with column2:
            st.write("")
        with column3:
            st.write("")
        with column4:
            st.markdown("[![Foo](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/reynaldihutapea/)")
        with column5:
            st.write("")
        with column6:
            st.write("")
        with column7:
            st.write("")

        st.markdown("<h1 style='text-align: center;</h1> [![Foo](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/reynaldihutapea/)", unsafe_allow_html=True)
        html_temp_home1 = """<div style="padding:10px">
                                        <h4 style="color:white;text-align:center;font-size:18px;">
                                        Also check on my other project 😉✨</h4>
                                        </div>
                                        </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

  
        column1, column2, column3, column4, column5, column6, column7= st.columns(7)
        with column1:
            st.write("")
        with column2:
            st.write("")
        with column3:
            st.write("")
        with column4:
            st.markdown("[![Foo](https://img.icons8.com/fluency/2x/link.png)](https://linktr.ee/reynaldigev)")
        with column5:
            st.write("")
        with column6:
            st.write("")
        with column7:
            st.write("")

        st_lottie(lottie_ty, key="ty")
        
    st.sidebar.image(bot_image, use_column_width='auto')
  
if __name__ == "__main__":
    main()
