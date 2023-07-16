import streamlit as st
import streamlit.components.v1 as components
import cv2
import logging as log
import datetime as dt
from time import sleep

# To learn about haar cascade : https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
# to learn more about Streamlit : https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Streamlit begins

# Title
st.title("Face Detection Through Opencv Using Streamlit")
st.header("Streamlit")

# Render the h1 block, contained in a frame of size 700x100.
components.html("<html><body><h3>Streamlit is an open-source Python library that makes it easy to "
                "create and share beautiful, custom web apps for machine learning and data science. "
                "In just a few minutes you can build and deploy powerful data apps.</h3></body></html>"
                , width=700, height=100)
# Another way to use html in streamlit.
st.markdown("<html><I>Make sure that you have Python 3.6 - Python 3.8 installed.</I><br></html>",
            unsafe_allow_html=True)

# Building a sidebar
st.sidebar.subheader("Details of the person")
t1 = st.sidebar.text_input("Name of the Person 1")
s1 = st.sidebar.slider("Age of the person 1")

st.sidebar.markdown("---")

st.sidebar.subheader("Details of the person")
t2 = st.sidebar.text_input("Name of the Person 2")
s2 = st.sidebar.slider("Age of the person 2")

st.write("Name: ",t1)
st.write("Age: ", s1) # taking data from the sidebar
st.write("Name: ",t2)
st.write("Age: ", s2) # taking data from the sidebar

# Some other functions.
# st.map()
# st.radio
# st.checkbox
# st.pyplot
# st.images


# Selection box
# first argument takes the titleof the selectionbox second argument takes options
How_is_streamlit = st.selectbox("likings: ",['Very much', 'So so', 'Boring', 'Useless'])
st.write("Your review is: ", How_is_streamlit)

st.markdown(f'<hr style="height:2px;border:none;color:#333;background-color:#333;" />', unsafe_allow_html=True)
# Face Detection works

st.header("Opencv - Detection of Faces")
st.subheader("Understanding the methods we have used - what is Haar Cascades? ")

st.write("Object Detection using Haar feature-based cascade classifiers is an effective object detection "
         "method proposed by Paul Viola and Michael Jones in their paper, 'Rapid Object Detection using a "
         "Boosted Cascade of Simple Features' in 2001. It is a machine learning based approach where a "
         "cascade function is trained from a lot of positive and negative images. It is then used to "
         "detect objects in other images.")

# Accessign haar files: https://github.com/opencv/opencv/tree/master/data/haarcascades
components.html('<html><body>Note: The haar cascade files can be downloaded from the '
            '<a href = "https://github.com/opencv/opencv/tree/master/data/haarcascades">'
            'OpenCV Github repository</a></body></html>')

st.write("For Example, if you go to the github page of haarcascade you will see that there is a particular"
         " xml file containing the feature set to detect the full body, lower-body, eye, frontal-face and "
         "so on.")

st.subheader("Starting with our Face Detection app building")
st.write("We will use the cv::CascadeClassifier class to detect objects in a video stream. Particularly,"
         " we will use the functions:")
st.write("-> cv::CascadeClassifier::load to load a .xml classifier file. It can be either a Haar or a LBP "
         "classifier")
st.write("-> cv::CascadeClassifier::detectMultiScale to perform the detection.")

st.subheader("Step 1:")
st.write("We use the CascadeClassifier function of OpenCV to point to the location where we have "
         "stored the XML file, haarcascade_frontalface_default.xml in our case. I have downloaded "
         "the xml file to my local and used the path of my machine")
# Highlighting Information
st.info('cascPath = "haarcascade_frontalface_default.xml"')
st.info('faceCascade = cv2.CascadeClassifier(cascPath)')

st.subheader("Step 2:")
st.write("Now the 2nd step is to load the frames and convert it into gray-scale."
         " I want to tell you the reason why we are converting the image to grayscale here."
         "Generally the images that we see are in the form of RGB channel(Red, Green, Blue). So, "
         "when OpenCV reads the RGB image, it usually stores the image in BGR (Blue, Green, Red) "
         "channel. For the purposes of image recognition, we need to convert this BGR channel to gray "
         "channel. The reason for this is gray channel is easy to process and is computationally less "
         "intensive as it contains only 1-channel of black-white.")

st.info("ret, frame = video_capture.read()")
st.info("gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)")

st.subheader("Step 3:")
st.write("Now after converting the image from RGB to Gray, we will now try to locate the exact"
         " features in our face. Let’s see how we could implement that in code.")
st.info("""faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )""")

st.write(" we are using an inbuilt function with it called the detectMultiScale."
         "This function will help us to find the features/locations of the new image. The way it"
         " does is, it will use all the features from the faceCascade object to detect the "
         "features of the new image.")
st.write("The parameters that we will pass to this function are:")
st.write("-> The gray scale variable — gray in our case")
st.write("-> scaleFactor — Parameter specifying how much the image size is reduced at each image scale")
st.write("-> minNeighbors — Parameter specifying how many neighbors each candidate rectangle should have "
         "to retain it. ")
st.write("-> minSize - Minimum possible object size. Objects smaller than that are ignored.")

st.subheader("Step 4:")
st.write("From the above step, the function detectMultiScale returns 4 values — x-coordinate, "
         "y-coordinate, width(w) and height(h) of the detected feature of the face. Based on these "
         "4 values we will draw a rectangle around the face.")
st.info("for (x, y, w, h) in faces:")
st.info("   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)")

st.subheader("Step 5:")
st.write("Lastly we are waiting till the user inputs 'q', then exit all processes, releasing all captures")
st.info("if cv2.waitKey(1) & 0xFF == ord('q'): break")

st.header("Les gooo")
if st.button("Can I detect your face ?"):
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()