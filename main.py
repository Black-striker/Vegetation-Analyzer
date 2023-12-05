import calendar  # Core Python Module
from datetime import datetime  # Core Python Module
from google.cloud import firestore
import plotly.graph_objects as go  # pip install plotly
import streamlit as st  # pip install streamlit
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

#cred = credentials.Certificate(service_account_key)
#FIREBASE = firebase_admin.initialize_app(cred)

# Authenticate to Firestore with the JSON account key.
db = firestore.Client.from_service_account_json("D:\Project_Lawn_Drone\irestore-key.json")

# Create a reference to the Google post.
doc_ref = db.collection("test").document("test1")


def label(image, text):
    """
    Labels the given image with the given text
    """
    return cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,255)

def disp_multiple(im1, im2, im3, im4):
    """
    Combines four images for display.

    """
    height, width = im1.shape

    combined = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)

    combined[0:height, 0:width, :] = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    combined[height:, 0:width, :] = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    combined[0:height, width:, :] = cv2.cvtColor(im3, cv2.COLOR_GRAY2RGB)
    combined[height:, width:, :] = cv2.cvtColor(im4, cv2.COLOR_GRAY2RGB)

    return combined

def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def histogram(ndvi):
    #Histogram of NDVI
    hist = cv2.calcHist([ndvi],[0],None,[256],[0,256])
    intensity_values = np.array([x for x in range(hist.shape[0])])
    return hist, intensity_values


def main_loop():

    # -------------- SETTINGS --------------
    page_title = "VEGETATION ANALYSER"
    page_icon = ":four_leaf_clover:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    # --------------------------------------

    st.set_page_config(page_title=page_title, page_icon=page_icon)
    st.title(page_title)


    # --- DROP DOWN VALUES FOR SELECTING THE PERIOD ---
    years = [datetime.today().year, datetime.today().year - 1, datetime.today().year - 2, datetime.today().year - 3, datetime.today().year - 4, datetime.today().year - 5, datetime.today().year - 6, datetime.today().year - 7, datetime.today().year - 8, datetime.today().year - 9, datetime.today().year - 10]
    months = list(calendar.month_name[1:])

    st.subheader("This application allows you to detect the health of crops from an image", divider='gray')

    new_title = '<p style="font-family:candara; font-size: 20px;"> The normalized difference vegetation index (NDVI) is a widely-used metric for quantifying the health and density of vegetation using sensor data. It is calculated from spectrometric data at two specific bands: red and near-infrared. The spectrometric data is usually sourced from remote sensors, such as satellites. The metric is popular in industry because of its accuracy. It has a high correlation with the true state of vegetation on the ground. The index is easy to interpret: NDVI will be a value between -1 and 1. An area with nothing growing in it will have an NDVI of zero. NDVI will increase in proportion to vegetation growth. An area with dense, healthy vegetation will have an NDVI of one. NDVI values less than 0 suggest a lack of dry land. An ocean will yield an NDVI of -1.</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    n_title = '<p style="font-family:candara; font-size: 20px;">This project uses the ndvi algorithm to detect areas of a picture with healthy vegetation and areas of a picture with poor vegetation and highlights them using image processing. It uses the color intensities of the picture to make the perfect vegetation analysis, which can be useful for businesses and farming. </p>'
    st.markdown(n_title, unsafe_allow_html=True)

    logopic = cv2.imread("D:\Project_Lawn_Drone\logo.png",cv2.IMREAD_COLOR)
    logo = st.sidebar.image([logopic])
    brightness = st.sidebar.slider("Brightness", min_value=0, max_value=50)
    contrast = st.sidebar.slider("Contrast", min_value=-5, max_value=5, value=1)
    st.text("")
    st.subheader("Vegetation Analysis")
    magnitude_spectrum = st.sidebar.checkbox('Show Magnitude Spectrum')
    show_histogram = st.sidebar.checkbox('Show Histogram')
    show_comparision = st.sidebar.checkbox('Compare')


    # --- NAVIGATION MENU ---
    selected = option_menu(
        menu_title=None,
        options=["Vegetation Analyser","Data Entry", "Data Visualization"],
        icons=["pencil-fill", "bar-chart-fill"],  # https://icons.getbootstrap.com/
        orientation="horizontal",
    )

    if selected == "Vegetation Analyser":
        image_file = st.file_uploader("Upload Your Image Here: ", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            original_image = Image.open(image_file)
            original_image = np.array(original_image)

            # Get the individual colour components of the image
            b, g, r = cv2.split(original_image)

            # Calculate the NDVI

            # Bottom of fraction
            bottom = (r.astype(float) + b.astype(float))
            bottom[bottom == 0] = 0.01  # Make sure we don't divide by zero!

            ndvi = (r.astype(float) - b) / bottom
            ndvi = contrast_stretch(ndvi)
            ndvi = ndvi.astype(np.uint8)

            # Do the labelling
            label(b, 'Blue')
            label(g, 'Green')
            label(r, 'NIR')
            label(ndvi, 'NDVI')

            # Combine ready for display
            combine = disp_multiple(b, g, r, ndvi)
            nnndvi = cv2.merge([ndvi,ndvi,r])
            nnndvi = cv2.convertScaleAbs(nnndvi,alpha=contrast,beta=brightness)

            # Percentage of Healthy Vegetation
            histaaaa = cv2.calcHist([ndvi],[0],None,[256],[0,256])
            intensity_valuesaa = np.array([x for x in range(histaaaa.shape[0])])
            percentile = np.count_nonzero((ndvi > 130))
            percentilediv = np.count_nonzero((ndvi>0))
            percent = (percentile/percentilediv)*100


            st.subheader("Original Image vs Processed Image")
            st.image([original_image, nnndvi])
            st.text("")
            st.text("")
            sign = "%"
            col1, col2 = st.columns(2)
            col1.metric("Percentage of healthy Vegetation", f"{percent} {sign}")

            if magnitude_spectrum:
                st.text("")
                st.text("")
                st.subheader("Magnitude Spectrum")
                imagee = cv2.imread("D:\Project_Lawn_Drone\ibar.jpg",cv2.IMREAD_COLOR)
                nimagee = cv2.cvtColor(imagee,cv2.COLOR_BGR2RGB)
                st.image([nimagee])
                st.text("Unhealthy Vegetation    |     Moderate Health     |     Healthy Vegetation")
                
                
            
            if show_histogram:
                hist, intensity_values = histogram(ndvi)
                st.text(" ")
                st.text(" ")
                st.subheader("The Health of Vegetation")
                st.bar_chart(hist)

            if show_comparision:
                st.text("")
                st.subheader("Color Intensity of the Image")
                st.image([combine])
            
            namee = '<p style="font-family:arial; font-size: 15px; font-weight:1000">SHERWIN SMITH</p>'
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.markdown(namee, unsafe_allow_html=True)    


    # --- INPUT & SAVE PERIODS ---
    if selected == "Data Entry":
        st.header(f"Data Entry")
        optionss = st.selectbox("Please Select an Option",("Create Data", "Update Data", "Delete Data"))
        if optionss == "Create Data":
            with st.form("entry_form", clear_on_submit=True):
                col1, col2, col3 = st.columns(3)
                record = col1.number_input("Record No", format="%i", step=1, key="record")
                month = col2.selectbox("Select Month:", months, key="month")
                year = col3.selectbox("Select Year:", years, key="year")
                year = str(year)
                record = str(record)

                #st.date_input("Select Date & Time", value="today", format="DD/MM/YYYY", label_visibility="visible")
                "---"
                with st.expander("Percentage"):
                    percent = st.number_input("Enter Percentage of Healthy Vegetation", min_value=0, max_value=100, format="%i", step=10, key="percent")

                with st.expander("Comment"):
                    comment = st.text_area("", placeholder="Enter a comment here ...")

                    "---"
                
                submitted = st.form_submit_button("Save Data")
                if submitted:
                    doc_ref = db.collection("test").document(record)
                    doc_ref.set({
                        "Date": month+"_"+year,
                        "Percentage": percent
                    })
                    st.success("Data Saved")

        if optionss == "Delete Data":
            with st.form("entry_form", clear_on_submit=True):
                col1, col2, col3 = st.columns(3)
                record = col1.number_input("Record No", format="%i", step=1, key="record")
                month = col2.selectbox("Select Month:", months, key="month")
                year = col3.selectbox("Select Year:", years, key="year")
                record = str(record)
                submitted = st.form_submit_button("Delete Data")
                if submitted:
                    doc_ref = db.collection("test").document(record).delete()
                    st.success("Data Deleted")

        if optionss == "Update Data":
            with st.form("entry_form", clear_on_submit=True):
                col1, col2, col3 = st.columns(3)
                record = col1.number_input("Record No", format="%i", step=1, key="record")
                record = str(record)

                coloum2, coloum3 = st.columns(2)
                month = coloum2.selectbox("Select New Month", months, key="month")
                year = coloum3.selectbox("Select New Year:", years, key="year")
                year = str(year)

                with st.expander("Updated Percentage"):
                    percent = st.number_input("Enter Percentage of Healthy Vegetation", min_value=0, max_value=100, format="%i", step=10, key="percent")

                with st.expander("Updated Comment"):
                    comment = st.text_area("", placeholder="Enter a comment here ...")
                
                submitted = st.form_submit_button("Update Data")
                if submitted:
                    doc_ref = db.collection("test").document(record)
                    doc_ref.update({
                        "Date": month+"_"+year,
                        "Percentage": percent
                    })
                    st.success("Data Updated")

        

    # --- PLOT PERIODS ---
    if selected == "Data Visualization":
        st.header("Data Visualization")
        

        percentarray = []
        rec = []
        datearray = []
        dataa_ref = db.collection("test").document("test1")
        doc = dataa_ref.get()
        posts_ref = db.collection("test")
        for doc in posts_ref.stream():
            reco = doc.id         
            rec.append(reco)
            perc = u'{}'.format(doc.to_dict()['Percentage'])
            perc = int(perc)
            percentarray.append(perc)
            datee = u'{}'.format(doc.to_dict()['Date'])
            datearray.append(datee)

        with st.form("retrieve_form", clear_on_submit=True):
            coll1, coll2, coll3 = st.columns(3)
            record = coll1.number_input("Record No", format="%i", step=1, key="record")
            month = coll2.selectbox("Select Month:", months, key="month")
            year = coll3.selectbox("Select Year:", years, key="year")
            year = str(year)
            record = str(record)
            st.text("")
            st.text("")
            submitted = st.form_submit_button("Retrieve Data")
            if submitted:
                roww = rec.index(record)
                if roww is None:
                    col1, col2 = st.columns(2)
                    col1.metric("Percentage of healthy Vegetation", f"Not Available")
                    return
                else:
                    sign = "%"
                    percent = percentarray[roww]
                    col1, col2 = st.columns(2)
                    col1.metric("Percentage of healthy Vegetation", f"{percent} {sign}")
        

        #for i in range (0,5):
            #st.write("", rec[i])
            #st.write("", percentarray[i])
            #st.write("", datearray[i])
        #record = 2
        
        data = pd.DataFrame({
            'Record No': rec,
            'Percent': percentarray
        })


        st.write("")
        st.write("")
        st.line_chart(data, x='Record No', y='Percent')


        # with st.form("saved_periods"):
            
        # Then get the data at that reference.
            


if __name__ == '__main__':
    main_loop()


