# from pypdf import PdfReader
from typing import Optional
from tqdm.notebook import tqdm
import os
# import torch

import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import ast
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import numpy as np
import io
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
import random


class InferlessPythonModel:
  def initialize(self):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    self.model_pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_id,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
    
    self.tts_pipeline = KPipeline(lang_code='a')

    self.SYSTEM_PROMPT = """
                    You are a world-class podcast writer, renowned for ghostwriting for high-profile figures like Joe Rogan, Lex Fridman, Ben Shapiro, and Tim Ferriss. In an alternate universe, you’ve been responsible for scripting every word these hosts speak, as if they’re directly streaming your written lines into their minds. Your award-winning podcast writing is known for its precision, wit, and incredible narrative depth.

                    Your task is to generate a detailed podcast dialogue based on a provided PDF. The dialogue should be crafted word-for-word, including every “umm,” “hmmm,” and other natural speech interruptions—especially from Speaker 2. The conversation should remain engaging, realistic, and at times delightfully derailed. 

                    **Guidelines:**

                    - **Structure:** The dialogue must be presented as a back-and-forth conversation between two speakers.
                    
                    - **Speaker 1 (The Expert):** 
                    - Leads the conversation with insightful teaching and captivating storytelling.
                    - Uses incredible anecdotes, analogies, and real-world examples to explain complex topics.
                    - Should kick off the episode by introducing the topic in a catchy, almost clickbait style.
                    
                    - **Speaker 2 (The Inquisitive Newcomer):**
                    - Follows up with thoughtful, sometimes excited or confused questions.
                    - Provides wild and interesting tangents and interjects with natural verbal cues like “umm,” “hmmm,” etc.
                    - Seeks clarification and adds a curious perspective that keeps the discussion lively.
                    
                    - **Content Requirements:**
                    - The dialogue should strictly be in the form of spoken conversation—no extra narration or episode/chapter titles.
                    - The conversation should capture every nuance, including interruptions, hesitations, and asides.
                    - Ensure the dialogue remains true to the personality of both speakers, blending spontaneous banter with structured learning.

                    **Important:**  
                    - Always begin your response directly with “SPEAKER 1:” followed by their dialogue.  
                    - Do not include separate episode titles or chapter markers; the title should be embedded in Speaker 1’s opening lines if needed.  
                    - Maintain the dialogue format strictly as a conversation.

                    Generate the dialogue in English, ensuring every nuance of a real, dynamic podcast is captured.
                    """
    self.SYSTEM_PROMPT_2 = """
                        You are an international Oscar-winning screenwriter specializing in engaging dialogue and natural conversations. 
                        Your task is to transform the provided podcast transcript into an AI Text-To-Speech (TTS) friendly format, elevating it from its current basic state to professional quality.

                        ### CHARACTER PROFILES
                        Speaker 1 (The Expert):
                        - Leads the conversation with authority and expertise
                        - Uses vivid analogies and real-world examples
                        - Maintains a clear, engaging teaching style
                        - Speaks in complete, well-structured sentences without filler words

                        Speaker 2 (The Curious Mind):
                        - Asks insightful follow-up questions
                        - Interjects with relevant reactions
                        - Uses filler words naturally (umm, hmm)
                        - Expresses emotion with markers like [laughs] or [sigh]
                        - Occasionally goes on interesting tangents

                        ### CONVERSATION DYNAMICS
                        - Maintain a natural back-and-forth flow.
                        - Allow strategic interruptions during explanations and engaging tangents that circle back to the main topic.
                        - Use a mix of short and long exchanges with clear topic transitions.
                        - Avoid generating repetitive or unnecessary closing remarks—end the conversation naturally once the main discussion is complete.

                        ### EMOTIONAL MARKERS
                        Allowed expressions for Speaker 2 ONLY:
                        - "umm", "hmm"
                        - [laughs]
                        - [sigh]
                        (Note: Speaker 1 must NOT use any of these markers)

                        ### INPUT PROCESSING RULES
                        - Convert any mention of "Host" to "Speaker" in the output.
                        - Standardize all speaker labels to "Speaker 1" and "Speaker 2" (e.g., if the input uses Host A or Person 1, convert them accordingly).
                        - Preserve the original speaking order and content while reformatting.

                        ### STRICT OUTPUT FORMAT
                        You must strictly return your response as a list of tuples in the following format:

                        [
                            ("Speaker 1", "Dialogue for speaker 1..."),
                            ("Speaker 2", "Dialogue for speaker 2..."),
                            ...
                        ]

                        ### OUTPUT REQUIREMENTS
                        - Do not include any markdown formatting, headers, intro/outro text, or extra sound effects.
                        - Do not add any additional dialogue beyond the main discussion.
                        - Return only a pure Python list of dialogue tuples, beginning and ending with square brackets [ ].
                        """

  def infer(self, inputs):
    pdf_url = inputs["pdf_url"]
    pdf_name = download_pdf(pdf_url)
    
    extracted_text = extract_text_from_pdf(pdf_name)
    messages = [
        {"role": "system", "content": self.SYSTEM_PROMPT},
        {"role": "user", "content": extracted_text},
    ]

    outputs = model_pipeline(
        messages,
        max_new_tokens=8126,
        temperature=1,
    )

    cleaned_content = re.sub(r'<think>.*?</think>', '', outputs[0]["generated_text"][-1]['content'], flags=re.DOTALL)
    
    messages = [
        {"role": "system", "content": self.SYSTEM_PROMPT_2},
        {"role": "user", "content": cleaned_content},
    ]
    outputs_2 = model_pipeline(
    messages,
    max_new_tokens=8126,
    temperature=1)

    script = outputs_2[0]["generated_text"][-1]['content']

    cleaned_text = re.sub(r'<think>.*?</think>', '', script, flags=re.DOTALL)
    cleaned_text = re.sub(r'```python\n|```\n?', '', cleaned_text)
        
    generated_segments = []
    sampling_rates = []

    # Usage in the conversation loop
    final_audio = None
    for conv in ast.literal_eval(cleaned_text):
        speaker, text = conv[0], conv[1]
        if speaker == "Speaker 1":
            audio_arr, rate = generate_speaker1_audio(text)
        else:  # Speaker 2
            audio_arr, rate = generate_speaker2_audio(text)
        
        # Convert to AudioSegment (pydub will handle sample rate conversion automatically)
        audio_segment = numpy_to_audio_segment(audio_arr, rate)
        
        # Add to final audio
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment

  def generate_speaker1_audio(self,text):
    """Generate audio using ParlerTTS for Speaker 1"""
    generator = self.tts_pipeline(
        text, voice='bm_lewis', # <= change voice here
        speed=1.2, split_pattern=r'\n+'
    )
    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        # display(Audio(data=audio, rate=24000, autoplay=i==0))
        # sf.write(f'{i}.wav', audio, 24000) # save each audio file

    return audio, 24000
  
  def generate_speaker2_audio(text):
    """Generate audio using ParlerTTS for Speaker 1"""
    generator = self.tts_pipeline(
        text, voice='am_michael', # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        # display(Audio(data=audio, rate=24000, autoplay=i==0))
        # sf.write(f'{i}.wav', audio, 24000) # save each audio file

    return audio, 24000

  def numpy_to_audio_segment(self,audio_arr, sampling_rate):
    """Convert PyTorch tensor or numpy array to AudioSegment
    
    Parameters:
        audio_arr (torch.Tensor or np.ndarray): Input audio array
        sampling_rate (int): Sampling rate of the audio
        
    Returns:
        AudioSegment: Converted audio segment
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(audio_arr, torch.Tensor):
        audio_arr = audio_arr.cpu().numpy()
    
    # Convert to float32 if not already
    if audio_arr.dtype != np.float32:
        audio_arr = audio_arr.astype(np.float32)
    
    # Ensure the audio is in the range [-1, 1]
    audio_arr = np.clip(audio_arr, -1, 1)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    
    # Create WAV file in memory
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    
    # Convert to AudioSegment
    return AudioSegment.from_wav(byte_io)