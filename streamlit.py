import azure.ai.vision as visionsdk
import streamlit as st
from streamlit_lottie import st_lottie , st_lottie_spinner
import json
# Azure Vision API credentials
key = "a85b1db65bb24616ba084b707f1a1ddf"
endpoint = "https://shreyaobj.cognitiveservices.azure.com/"
import io

def load_lottiefile(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    with st_lottie_spinner(load_lottiefile("waiting.json"),
                height=500, width='100%'):
        
            result = image_analyzer.analyze()

    # Display analysis results
            display_results(result)

def display_results(result):
    if result.reason == visionsdk.ImageAnalysisResultReason.ANALYZED:
        if result.objects is not None:
            st.subheader("Objects:")
            for obj in result.objects:
                st.write(f"- '{obj.name}', {obj.bounding_box}, Confidence: {obj.confidence:.4f}")

        if result.tags is not None:
            st.subheader("Tags:")
            for tag in result.tags:
                st.write(f"- '{tag.name}', Confidence: {tag.confidence:.4f}")

        if result.people is not None:
            st.subheader("People:")
            for person in result.people:
                st.write(f"- {person.bounding_box}, Confidence: {person.confidence:.4f}")

        if result.text is not None:
            st.subheader("Text:")
            for line in result.text.lines:
                points_string = "{" + ", ".join([str(int(point)) for point in line.bounding_polygon]) + "}"
                st.write(f"- Line: '{line.content}', Bounding polygon: {points_string}")
                for word in line.words:
                    points_string = "{" + ", ".join([str(int(point)) for point in word.bounding_polygon]) + "}"
                    st.write(f"  - Word: '{word.content}', Bounding polygon: {points_string}, Confidence: {word.confidence:.4f}")
    else:
        error_details = visionsdk.ImageAnalysisErrorDetails.from_result(result)
        st.write("Analysis failed.")
        st.write(f"Error reason: {error_details.reason}")
        st.write(f"Error code: {error_details.error_code}")
        st.write(f"Error message: {error_details.message}")
        st.write("Did you set the computer vision endpoint and key?")

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