import numpy as np
import os
import cv2
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
from glob import glob
st.title("Food Adulteration Detector")
img1 = st.file_uploader("Choose an image...", type="tif")
if img1 is not None: 
    fn = Image.open(img1)
    imageNo1 = cv2.imread('a6.tif', 0)
    cv2.ocl.setUseOpenCL(False)
    imageNo2 = cv2.imread('a6.tif',0) 
    imageNo1 = imageNo1[200:600, 200:1000] 
    orb = cv2.ORB_create()
    keypoint1, descriptor1 = orb.detectAndCompute(imageNo1,None)
    keypoint2, descriptor2 = orb.detectAndCompute(imageNo2,None)
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matchResults = bfMatcher.match(descriptor1,descriptor2)
    matchResults = sorted(matchResults, key = lambda x:x.distance)
    resultImage = cv2.drawMatches(imageNo1,keypoint1,imageNo2,keypoint2,matchResults[:100], None, flags=2)
    plt.imshow(resultImage),plt.show()
    st.image(resultImage)
    if (matchResults[171:200]):
        st.write("90% unadulterated")
    elif (matchResults[152:170]):
        st.write("80% unadulterated")
    elif (matchResults[131:151]):
        st.write("70% unadulterated")
    elif (matchResults[120:130]):
        st.write("60% unadulterated")
    elif (matchResults[0:119]):
        st.write("Adulterated")