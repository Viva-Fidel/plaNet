import cv2
import pandas as pd
import streamlit as st
from PIL import Image

import zipfile

from recognition import Recognition


# @st.experimental_singleton
class Main:

    # @st.cache
    # def load_image(new_image):
    # im = Image.open(new_image)
    # return im

    # @st.cache
    # def convert_df(df):
    # return df.to_csv().encode('utf-8')

    def create_zip(result, file_name):
        with zipfile.ZipFile('result.zip', 'a') as final_zip:
            cv2.imwrite(file_name, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            final_zip.write(file_name)

    st.title('PlaNet - neural web for plant detection and progress tracking')
    activities = ['Detection', 'Updates', 'To Do']
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

            result_img = Recognition(cv2.cvtColor(cv2.imread(uploaded_file.name), cv2.COLOR_BGR2RGB), uploaded_file.name)
            result = result_img.do_detection()

            st.image(result)
            create_zip(result, uploaded_file.name)

        plant_df = pd.DataFrame(Recognition.plant_data, columns=['File_name', 'Plant_type', 'Color', 'Leaves_area'])
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

    elif choice == 'Updates':
        st.subheader('Updates')
        st.write('02.09.2022. Rewrote code using OOP. Fixed bugs. Added To Do page')
        st.write('03.09.2022. Fixed bugs. Refactored code')

    elif choice == 'To Do':
        st.subheader('To Do')
        st.write('Fix bugs')
        st.write('NaN if no detection')
        st.write('Graphs visualization')
        st.write('Configure Cache')
        st.write('New Features')
        st.write('Beautiful readme on GitHub')
        st.write('Plant type recognition')

if __name__ == '__main__':
    website = Main()
