import cv2
import numpy as np
import zipfile
from utils import *
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance

plant_data = []


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


def plant_detection(file_name):
    current_plant = []
    class_ids = []
    confidences = []
    boxes = []

    # upload file and process
    img = cv2.imread(file_name)
    current_plant.append(file_name)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    PlantNet.setInput(blob)
    outs = PlantNet.forward(output_layers)

    # detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.75:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # draw bounding box
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, label, (x, y - 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 51, 51), 2)
    current_plant.append(label)

    # making mask for plant
    cropped_image = img[y:y + h, x:x + w]
    img_plant = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_plant)
    img_plant = cv2.subtract(B, A)

    _, thresh1 = cv2.threshold(img_plant, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(thresh1, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    contours_plant, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_plant = 0
    for cnt in contours_plant:
        if cv2.contourArea(cnt) > 0:
            area_plant += cv2.contourArea(cnt)

    # making mask for blue square 1x1
    img_square = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(img_square, lower_blue, upper_blue)
    contours_square, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_square = 0
    for cnt in contours_square:
        area_square += cv2.contourArea(cnt)

    # area calculation
    sqrcm_plant = round(area_plant / area_square, 2)
    sqrcm_text = f'''{file_name}
    Area of leaves: {sqrcm_plant}'''
    current_plant.append(sqrcm_plant)

    # adding text
    cv2.putText(img, sqrcm_text, (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 51, 51), 3, 2)
    plant_data.append(current_plant)
    return img


def main():
    st.title('plaNet - neural web for plant detection')

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Detection':
        st.subheader('Plant Detection')

        uploaded_files = st.file_uploader('Upload images', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
        with zipfile.ZipFile('result.zip', 'w') as final_zip:
            pass

        for uploaded_file in uploaded_files:
            new_image = Image.open(uploaded_file)
            new_image.save(uploaded_file.name)
            st.image(new_image)
            result_img = plant_detection(uploaded_file.name)
            with zipfile.ZipFile('result.zip', 'a') as final_zip:
                zip_file_name = uploaded_file.name
                zip_image = result_img
                cv2.imwrite(zip_file_name, zip_image)
                final_zip.write(zip_file_name)

            st.image(result_img)

        plant_df = pd.DataFrame(plant_data, columns=['File_name', 'Plant_type', 'Leaves_area'])
        csv = convert_df(plant_df)

        #final_zip = 'result.zip'

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='plant_data.csv',
            mime='text/csv',
        )

        st.download_button(
            label="Download ZIP file with results",
            data=open('result.zip', 'rb').read(),
            file_name='result.zip',
            mime='application/zip')


    elif choice == 'About':
        st.subheader('About')


if __name__ == '__main__':
    main()
