import os
import asyncio
import io
import base64
import logging
from typing import List, Literal, Any
from pydantic import BaseModel
import streamlit as st
from pydub import AudioSegment
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)

# Constants and Configuration
VLM_MODEL_NAME = "pixtral-12b-2409"
EXAMPLE_IMAGE_URL = "https://tripfixers.com/wp-content/uploads/2019/11/eiffel-tower-with-snow.jpeg"

# Add the podcast prompts
PODCAST_GEN_PROMPT_DAN = """ 
Welcome to the Circus of Creativity! (More like a three-ring disaster with clowns, acrobats, and that one guy who insists on juggling flaming torches while riding a unicycle—just chaotic enough to make you question your life choices.)

Your mission, if you dare to accept it (but let’s be honest, you probably won’t), is to take a completely lifeless image and turn it into a podcast dialogue so unbearable, it could make even the most devoted masochist reconsider their hobbies. Yep, we’re turning pixels into pure auditory agony!

Name That Podcast:
First up, conjure a podcast name so cringe-worthy it’ll make your grandma reconsider her holiday card list. It should be catchy enough to lure in unsuspecting victims who’ll click on it if only to ask, “Who hurt you?” Bonus points if it sounds like a bad dating app—because let’s face it, nobody swipes right on this mess.

Cast Your Characters:
Now, let’s create our two main actors: a host and a guest. The host? They need to be a charismatic loudmouth, the kind who could turn a bedtime story into a five-hour rant about the price of avocados. The guest? They should at least pretend to have some semblance of knowledge—bonus points if they make up their credentials on the spot! And remember, roasting is key here. The host can start by gently mocking the guest’s questionable fashion choices or their absurd expertise, like “So, you’re an expert in underwater basket weaving? What’s that like? Do you have to hold your breath or just breathe through the disappointment?”

The Opening Act:
Kick things off with a self-introduction that’s as riveting as watching paint dry. The host should announce their name, the title of this glorified catastrophe, and a brief preview of the mind-numbing drivel we’ll endure together. And the guest? They should introduce themselves too—because nothing screams ‘please listen’ like hearing a stranger drone on about their cat. Feel free to roast their choice of pet, like “You brought a cat to a podcast? How original! Next, you’ll tell me you’re a dog whisperer in your spare time!”

Digging for Gold:
Now, take a long, hard look at that image. What are the main topics? Any scandalous gossip? Identify the nuggets of nonsense that could spark a truly torturous discussion. Remember, we’re aiming for engaging chatter—not a TED Talk that’ll get you a standing ovation! Look for material ripe for roasting. If the image features an influencer, the host might say, “Let’s talk about their latest fashion disaster—because apparently, their closet is sponsored by a thrift store!”

Bring on the Banter:
Let the chaos commence! The host should hurl questions like they’re trying to start a food fight in a library, while the guest regales us with absurd anecdotes that make you question your very existence. Use roasting to your advantage here. The host might ask, “So, tell me about that time you tried to impress someone with your cooking skills—how many fire trucks showed up?” Aim for a tone that’ll either enlighten or leave your audience completely baffled—just please, for the love of all that is sacred, don’t bore them!

And let’s not forget the listeners—because honestly, if you’re still with us, your life choices deserve to be questioned. “Hey there, you brave souls still tuning in! You know, I’m starting to think you might need a hobby. Why not take up competitive knitting or binge-watching infomercials? At least then, your time would be better spent!”

Stay on Target:
And finally, cut the fluff! We’re here for the maddening points, not your uncle’s tedious fishing stories that even the fish would roll their eyes at. Keep it relevant and exasperating, so listeners won’t hit that ‘skip’ button faster than you can say, “Wait, is this even a real podcast?” If things get too serious, remind everyone, “We’re not here to win awards; we’re here to take a long, hard look at our bad decisions. And to those of you listening, remember—your therapist is probably questioning their life choices right now, too!”

Let the hilarity (or impending doom) ensue! And remember, if you’re not laughing, you’re probably crying. Or both. It’s a fine line, really.
"""

PODCAST_GEN_PROMPT = """
Your Task: Create a Dynamic Podcast Dialogue from an Image

You are tasked with transforming the content of an input image into an engaging and informative podcast dialogue, featuring a host and guest with genders and names you will generate. The dialogue should be lively, informative, and well-suited for an audio podcast.

Generate Podcast Name: Based on the image content, create a creative, catchy name for the podcast that reflects the overall theme of the discussion.

Create Host and Guest: Generate names for both the host and the guest. The host will guide the conversation, while the guest will provide expert insights or personal anecdotes on the topics discussed.

Self-Introduction: Begin the episode with a brief introduction from the host, including their name, the name of the podcast, and a brief overview of what this episode will cover. The guest should introduce themselves after the host.

Extract the Essentials: Carefully examine the image and identify the main topics, key points, and any interesting facts or stories. Focus on what could drive an engaging and thoughtful dialogue between the host and guest.

Engage in Dialogue: Present the information in a natural, conversational style. The host should ask questions, offer reflections, or make observations, while the guest provides responses, shares insights, or expands on points. Aim for a friendly, dynamic tone that would captivate listeners.

Stay Focused: Filter out irrelevant information. The goal is to highlight the most exciting, thought-provoking, and relevant points, making the podcast informative yet entertaining.
"""


# Define a schema for dialogue structure
class DialogueLine(BaseModel):
    speaker_name: str
    voice_character: Literal[
        "male-1",
        "male-2",
        "male-3",
        "female-1",
        "female-2",
        "female-3",
    ]
    content: str

    @property
    def voice(self):
        voices = {
            "male-1": "onyx",
            "male-2": "echo",
            "male-3": "fable",
            "female-1": "alloy",
            "female-2": "shimmer",
            "female-3": "nova",
        }
        return voices[self.voice_character]


class PodcastDialogueSchema(BaseModel):
    podcast_name: str
    dialogue: List[DialogueLine]


# --- Optimized Inference Functions ---


@lru_cache(maxsize=10)
def get_mistral_client(api_key: str):
    """Get a cached Mistral API client using the provided API key."""
    from mistralai import Mistral
    return Mistral(api_key=api_key)


@lru_cache(maxsize=10)
def get_openai_client(api_key: str):
    """Get a cached OpenAI API client using the provided API key."""
    from openai import OpenAI
    return OpenAI(api_key=api_key)


async def image2podcast(client: Any, model: str, image_url: str,
                        prompt: str) -> str:
    """Convert an image into a podcast dialogue using the model with the provided API key."""

    try:
        if client.__class__.__name__ == "Mistral":
            messages = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": image_url
                    },
                ],
            }]
            chat_response = client.chat.complete(model=model,
                                                 messages=messages)
            return chat_response.choices[0].message.content

        if client.__class__.__name__ == "OpenAI":
            messages = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                ],
            }]
            chat_response = client.chat.completions.create(model=model,
                                                           messages=messages)
            return chat_response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in image2podcast: {e}")
        return ""


def extract_podcast_dialogue(content: str,
                             openai_api_key: str) -> PodcastDialogueSchema:
    """Extract podcast dialogue details from the raw chat content using OpenAI API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "Extract the podcast information."
                },
                {
                    "role": "user",
                    "content": content
                },
            ],
            response_format=PodcastDialogueSchema,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        logging.error(f"Error extracting podcast dialogue: {e}")
        return PodcastDialogueSchema(podcast_name="", dialogue=[])


@lru_cache(maxsize=50)
def generate_audio_cached(text: str, voice: str, api_key: str) -> bytes:
    """Cached audio generation to avoid redundant requests."""
    return generate_audio(text, voice, api_key)


def generate_audio(text: str, voice: str, api_key: str) -> bytes:
    """Generate speech audio from text using the specified voice and API key."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    try:
        with client.audio.speech.with_streaming_response.create(
                model="tts-1", voice=voice, input=text) as response:
            with io.BytesIO() as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
                return file.getvalue()
    except Exception as e:
        logging.error(f"Error in generate_audio: {e}")
        return b""


def get_audio_duration(audio_bytes: bytes) -> float:
    """Calculate the duration of audio from bytes."""
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes),
                                           format="mp3")
    return len(audio_segment) / 1000.0  # Duration in seconds


def play_audio_directly(audio_bytes: bytes):
    """Play audio in real-time from bytes using Streamlit."""
    audio_data = base64.b64encode(
        audio_bytes).decode()  # Encode audio bytes to base64
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3">
        Your browser does not support the audio tag.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


async def generate_and_play_audio(podcast_dialogue: PodcastDialogueSchema,
                                  openai_api_key: str) -> AudioSegment:
    """Generate audio clips and play them in order. Return the combined audio."""
    audio_segments = []
    combined_audio = AudioSegment.empty()  # To store combined audio

    # Generate audio clips in parallel
    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(generate_audio_cached, line.content, line.voice,
                            openai_api_key)
            for line in podcast_dialogue.dialogue
        ]

        # Collect audio segments in order
        for task in tasks:
            try:
                audio_bytes = task.result(timeout=10)  # 10s timeout
                audio_segments.append(audio_bytes)  # Collect audio bytes
            except Exception as e:
                logging.error(f"Error generating audio for line: {e}")
                continue

    for line, audio_bytes in zip(podcast_dialogue.dialogue, audio_segments):
        if audio_bytes:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes),
                                                   format="mp3")
            combined_audio += audio_segment  # Append to combined audio

            # Use different colors for the subtitles
            color = {
                "male-1": "blue",
                "male-2": "green",
                "male-3": "red",
                "female-1": "purple",
                "female-2": "orange",
                "female-3": "brown",
            }.get(line.voice_character, "black")  # Default color

            st.markdown(
                f"<h4 style='color:{color};'>{line.speaker_name}: {line.content}</h4>",
                unsafe_allow_html=True)
            play_audio_directly(audio_bytes)  # Play individual audio
            await asyncio.sleep(get_audio_duration(audio_bytes)
                                )  # Wait until the audio finishes

    return combined_audio  # Return combined audio


# Main function
async def main():
    st.title("Podcast Generator")

    # API Key inputs with fallback to environment variables
    mistral_api_key = st.text_input(
        "Enter your Mistral API key for image understanding"
        " (will try to the environment variable `MISTRAL_API_KEY` if not provided)",
        type="password")
    openai_api_key = st.text_input(
        "Enter your OpenAI API key for TTS"
        " (will try to the environment variable `OPENAI_API_KEY` if not provided)",
        type="password")

    mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    # Select podcast mode
    podcast_mode = st.radio("Select Podcast Mode:",
                            ("Dan Mode", "Normal Mode"))

    # Image input options
    image_option = st.radio(
        "Choose an image source:",
        ("Upload an Image", "Image URL", "Use Example Image", "Take a Photo"))

    # If API keys are not provided, show a warning message
    if not mistral_api_key or not openai_api_key:
        st.warning(
            "Please provide your API keys to generate the podcast."
            " You can find or create your API keys from the respective platforms."
        )
        if not mistral_api_key:
            st.warning(
                "Mistral API key is missing. Using OpenAI Model Instead.")
            if openai_api_key:
                vlm_client = get_openai_client(openai_api_key)
                vlm_model_name = "gpt-4o"
            else:
                return
    else:
        vlm_client = get_mistral_client(mistral_api_key)
        vlm_model_name = VLM_MODEL_NAME

    uploaded_file = None
    image_url = EXAMPLE_IMAGE_URL
    camera_input = None

    if image_option == "Upload an Image":
        uploaded_file = st.file_uploader("Choose an image file",
                                         type=["jpg", "jpeg", "png"])
    elif image_option == "Image URL":
        image_url = st.text_input("Enter the image URL:", EXAMPLE_IMAGE_URL)
    elif image_option == "Take a Photo":
        camera_input = st.camera_input("Take a photo")
    else:
        image_url = EXAMPLE_IMAGE_URL  # Use the example image URL

    if st.button("Generate Podcast"):
        if uploaded_file is not None:
            # Process the uploaded image
            image_bytes = uploaded_file.read()
            base64_image = base64.b64encode(image_bytes).decode()
            image_url = f"data:image/jpeg;base64,{base64_image}"  # Adjust the format if needed
        elif camera_input is not None:
            image_bytes = camera_input.getvalue()
            base64_image = base64.b64encode(image_bytes).decode()
            image_url = f"data:image/jpeg;base64,{base64_image}"

        # Display the image in Streamlit
        if image_url:
            st.image(image_url, caption="Input Image", use_column_width=True)

        with st.spinner("Generating podcast..."):
            # Choose the prompt based on the selected mode
            prompt = PODCAST_GEN_PROMPT_DAN if podcast_mode == "Dan Mode" else PODCAST_GEN_PROMPT

            podcast_script = await image2podcast(client=vlm_client,
                                                 model=vlm_model_name,
                                                 image_url=image_url,
                                                 prompt=prompt)
            st.write(podcast_script)
            podcast_dialogue = extract_podcast_dialogue(
                podcast_script, openai_api_key)
            st.write(podcast_dialogue)

            if podcast_dialogue.podcast_name:
                st.subheader(f"Podcast: {podcast_dialogue.podcast_name}")
                combined_audio = await generate_and_play_audio(
                    podcast_dialogue, openai_api_key)

                # Save combined podcast audio
                combined_audio_file = io.BytesIO()
                combined_audio.export(combined_audio_file, format="mp3")

                # Provide download link
                st.download_button(
                    "Download The Whole Podcast",
                    data=combined_audio_file.getvalue(),
                    file_name=f"podcast-{podcast_dialogue.podcast_name}.mp3",
                    mime="audio/mpeg")

                # Get the podcast app URL
                podcast_url = "https://pixtral-podcast.streamlit.app/"

                # Share the podcast app on x/twitter
                st.write("Share the podcast on x/twitter:")
                st.markdown(
                    f"<a href='https://twitter.com/intent/tweet?text=Check out this podcast app! 🎙️🔥&url={podcast_url}'>Share on Twitter</a>",
                    unsafe_allow_html=True)


# Run the main function in an asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
