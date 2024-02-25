import azure.ai.vision as visionsdk
import streamlit as st
import cv2
import numpy as np

# Azure Vision API credentials
key = "a85b1db65bb24616ba084b707f1a1ddf"
endpoint = "https://shreyaobj.cognitiveservices.azure.com/"


def AnalyzeImage(filename: str, endpoint: str, key: str) -> None:
    service_options = visionsdk.VisionServiceOptions(endpoint, key)

    # Specify the image file to analyze
    vision_source = visionsdk.VisionSource(filename=filename)
    analysis_options = visionsdk.ImageAnalysisOptions()
    analysis_options.features = (
        visionsdk.ImageAnalysisFeature.OBJECTS |
        visionsdk.ImageAnalysisFeature.PEOPLE |
        visionsdk.ImageAnalysisFeature.TEXT |
        visionsdk.ImageAnalysisFeature.TAGS
    )

    image_analyzer = visionsdk.ImageAnalyzer(service_options, vision_source, analysis_options)

    st.write("Please wait for image analysis results...")
    result = image_analyzer.analyze()

    # Display analysis results
    display_results(result, filename)

def display_results(result, filename):
    image = cv2.imread(filename)
    image_copy = image.copy()
    if result.reason == visionsdk.ImageAnalysisResultReason.ANALYZED:
        if result.objects is not None:
            for obj in result.objects:
                # Draw rectangle around the object
                cv2.rectangle(image_copy, (obj.rectangle.x, obj.rectangle.y), (obj.rectangle.x + obj.rectangle.w, obj.rectangle.y + obj.rectangle.h), (0, 255, 0), 2)
                # Display object name and confidence
                cv2.putText(image_copy, f"{obj.name} ({obj.confidence:.2f})", (obj.rectangle.x, obj.rectangle.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the modified image
    cv2.imwrite("temp_image_with_boxes.jpg", image_copy)

    # Display the modified image on Streamlit webpage
    st.image("temp_image_with_boxes.jpg")

def main():
    st.title("Image Analysis with Azure Computer Vision")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.write(f"File path: {file_path}")
        AnalyzeImage(file_path, endpoint, key)

def save_uploaded_file(uploaded_file):
    # Use Streamlit's file_uploader utility to save the uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())
    return "temp_image.jpg"

if __name__ == "__main__":
    main()