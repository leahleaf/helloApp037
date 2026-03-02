# Program title: Storytelling App

# import part
import streamlit as st
from transformers import pipeline
from PIL import Image
import time

# --- Optimization: Cache models to prevent reloading on every user interaction ---
# This is a Streamlit best practice to improve user experience and avoid Out-Of-Memory errors
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_text2story_model():
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def load_text2audio_model():
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")


# --- function part ---

# img2text function
def img2text(url):
    # Load the cached model
    image_to_text_model = load_img2text_model()
    # Extract the generated text from the pipeline output
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story function
def text2story(text):
    # Load the cached model
    pipe = load_text2story_model()
    
    # Add a prompt starter to make it sound like a children's story
    prompt = f"Once upon a time, {text}. "
    
    # Add constraints to generate a narrative between 50 and 100 words (Assignment requirement)
    story_text = pipe(prompt, min_length=50, max_length=100, do_sample=True)[0]['generated_text']
    return story_text

# text2audio function
def text2audio(story_text):
    # Load the cached model
    pipe = load_text2audio_model()
    # Convert text to audio data
    audio_data = pipe(story_text)
    return audio_data


# --- main part ---
def main():
    # App page configuration
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")

    # App title and welcome text from the demo codes
    st.title("Streamlit Demo on Hugging Face")
    st.header("Turn Your Image to Audio Story")
    st.write("Welcome to a demo app showcasing basic Streamlit components!")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to the current working directory as bytes
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        # Display image with spinner to simulate a loading delay
        with st.spinner("Loading image..."):
            time.sleep(1)  # Simulate a delay for UI effect
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button interaction to trigger the AI pipeline
        if st.button("Click Me to Generate Story"):
            st.write("🎉 You clicked the button! Processing your magical story...")

            # Stage 1: Image to Text
            with st.spinner('Processing img2text...'):
                scenario = img2text(uploaded_file.name)
                st.success("Image caption generated successfully!")
                st.write(f"**Caption:** {scenario}")

            # Stage 2: Text to Story
            with st.spinner('Generating a story...'):
                story = text2story(scenario)
                st.success("Story generation completed!")
                st.write(f"**Story:** {story}")

            # Stage 3: Story to Audio data
            with st.spinner('Generating audio data...'):
                audio_data = text2audio(story)
                st.success("Audio generation completed!")

            # Play Audio directly using Streamlit
            st.write("🔊 **Listen to your story:**")
            
            # Get the audio array and sample rate from the pipeline dictionary output
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            
            st.audio(audio_array, sample_rate=sample_rate)

# Run the application
if __name__ == "__main__":
    main()
