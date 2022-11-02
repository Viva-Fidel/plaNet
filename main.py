import cv2
import pandas as pd
import streamlit as st
import zipfile

from PIL import Image

from advanced_plant_recognition import Advanced_plant_recognition
from basic_plant_recognition import Basic_plant_recognition
from root_recognition import Root_recognition

class Main:

    def create_zip(result, file_name):
        with zipfile.ZipFile('result.zip', 'a') as final_zip:
            cv2.imwrite(file_name, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            final_zip.write(file_name)

    st.title('PlaNet - neural web for plant detection and progress tracking')
    activities = ['Basic plant detection', 'Advanced plant detection', 'Plant root detection']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Basic plant detection':
        st.subheader('Basic plant detection')

        uploaded_files = st.file_uploader('Upload images', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

        with zipfile.ZipFile('result.zip', 'w') as final_zip:
            pass

        st.write("Progress bar")
        progress_bar = st.progress(0)
        bar_counter = 0

        for uploaded_file in uploaded_files:

            new_image = Image.open(uploaded_file)
            new_image.save(uploaded_file.name)

            st.image(new_image)

            result_img = Basic_plant_recognition(cv2.cvtColor(cv2.imread(uploaded_file.name), cv2.COLOR_BGR2HSV), uploaded_file.name)
            result = result_img.do_detection()
            result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

            st.image(result)
            create_zip(result, uploaded_file.name)

            bar_counter += 1 / len(uploaded_files)
            progress_bar.progress(bar_counter)

        plant_df = pd.DataFrame(Basic_plant_recognition.plant_data, columns=['File_name', 'Color', 'Leaves_area', 'Convex hull area'])
        csv = plant_df.to_csv().encode('utf-8')


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

    if choice == 'Advanced plant detection':
        st.subheader('Advanced plant detection')

        uploaded_files = st.file_uploader('Upload images', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

        with zipfile.ZipFile('result.zip', 'w') as final_zip:
            pass

        st.write("Progress bar")
        progress_bar = st.progress(0)
        bar_counter = 0

        for uploaded_file in uploaded_files:
            new_image = Image.open(uploaded_file)
            new_image.save(uploaded_file.name)

            st.image(new_image)

            result_img = Advanced_plant_recognition(cv2.cvtColor(cv2.imread(uploaded_file.name), cv2.COLOR_BGR2HSV),
                                                 uploaded_file.name)
            result = result_img.do_detection()
            result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

            st.image(result)
            create_zip(result, uploaded_file.name)

            bar_counter += 1 / len(uploaded_files)
            progress_bar.progress(bar_counter)

        plant_df = pd.DataFrame(Advanced_plant_recognition.plant_data,
                                columns=['File_name', 'Color', 'Leaves_area', 'Convex hull area'])
        csv = plant_df.to_csv().encode('utf-8')

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

    if choice == 'Plant root detection':
        st.subheader('Plant root detection')

        uploaded_files = st.file_uploader('Upload images', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

        with zipfile.ZipFile('result.zip', 'w') as final_zip:
            pass

        st.write("Progress bar")
        progress_bar = st.progress(0)
        bar_counter = 0

        for uploaded_file in uploaded_files:

            new_image = Image.open(uploaded_file)
            new_image.save(uploaded_file.name)

            st.image(new_image)

            result_img = Root_recognition(cv2.cvtColor(cv2.imread(uploaded_file.name), cv2.COLOR_BGR2HSV), uploaded_file.name)
            result = result_img.do_detection()
            result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

            st.image(result)
            create_zip(result, uploaded_file.name)

            bar_counter += 1 / len(uploaded_files)
            progress_bar.progress(bar_counter)

        plant_df = pd.DataFrame(Root_recognition.root_data, columns=['File_name', 'Root width', 'Root lenght'])
        csv = plant_df.to_csv().encode('utf-8')


        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='root_data.csv',
            mime='text/csv',
        )

        st.download_button(
            label="Download ZIP file with results",
            data=open('result.zip', 'rb').read(),
            file_name='result.zip',
            mime='application/zip')


if __name__ == '__main__':
    website = Main()
