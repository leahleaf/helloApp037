import streamlit as st
from transformers import pipeline

# 缓存 img2text 模型
@st.cache_resource
def get_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# 缓存 text2story 模型
@st.cache_resource
def get_text2story_model():
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

# 缓存 text2audio 模型
@st.cache_resource
def get_text2audio_model():
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")

# function part
def img2text(url):
    image_to_text_model = get_img2text_model()
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(text):
    pipe = get_text2story_model()
    story_text = pipe(text)[0]['generated_text']
    return story_text

def text2audio(story_text):
    pipe = get_text2audio_model()
    audio_data = pipe(story_text)
    return audio_data

def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(scenario)

        # Stage 2: Text to Story
        st.text('Generating a story...')
        story = text2story(scenario)
        st.write(story)

        # Stage 3: Story to Audio data
        st.text('Generating audio data...')
        audio_data = text2audio(story)

        # Play button
        if st.button("Play Audio"):
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            st.audio(audio_array, sample_rate=sample_rate)

if __name__ == "__main__":
    main()
