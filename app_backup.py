from typing import Optional
from tqdm.notebook import tqdm
import os
import torch
import base64
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import ast
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import io
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
import random

from utils import extract_list_of_tuples, download_pdf, set_seed, extract_text_from_pdf

import os
os.environ['PYTHONIOENCODING'] = 'UTF-8'


class InferlessPythonModel:
  def initialize(self):
    set_seed(seed=1526892603)
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    self.model_pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_id,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
    
    self.tts_pipeline = KPipeline(lang_code='a')

    self.CREATOR_PROMPT = """
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
    self.REFINE_PROMPT = """
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
        {"role": "system", "content": self.CREATOR_PROMPT},
        {"role": "user", "content": extracted_text},
    ]

    """outputs = self.model_pipeline(
        messages,
        max_new_tokens=8126,
        temperature=1,
    )

    cleaned_content = re.sub(r'<think>.*?</think>', '', outputs[0]["generated_text"][-1]['content'], flags=re.DOTALL)
    messages = [
        {"role": "system", "content": self.REFINE_PROMPT},
        {"role": "user", "content": cleaned_content},
    ]
    print("SECOND STAGE")
    outputs_refine = self.model_pipeline(
    messages,
    max_new_tokens=8126,
    temperature=1)

    outputs_refine_text = outputs_refine[0]["generated_text"][-1]['content']

    cleaned_outputs_text = re.sub(r'<think>.*?</think>', '', outputs_refine_text, flags=re.DOTALL)
    cleaned_outputs_text = re.sub(r'```python\n|```\n?', '', cleaned_outputs_text)
    lists_with_tuples = extract_list_of_tuples(cleaned_outputs_text)
    #print(lists_with_tuples.encode('utf-8', errors='replace').decode('utf-8'))
    generated_segments = []
    sampling_rates = []"""
    lists_with_tuples = [
    ("Speaker 1", "Welcome to the ultimate mind-blowing tech adventure! This is your captain, and today we're diving into the deep, dark, and slightly mysterious world of Meta AI's Llama 3.2! Prepare for your brain to be blown, folks!"),
    ("Speaker 2", "Oh my gosh! Wait, wait, wait! Llama 3.2? Is that like a giant stuffed animal with superpowers? Or is it some kinda secret AI project that's about to take over the world?"),
    ("Speaker 1", "Ah, my curious friend! Llama 3.2 is not an oversized plush toy, though it does have some pretty magical powers. It's an open-source AI model that lets developers fine-tune, distill, and deploy AI models anywhere. Think of it as a Swiss Army knife for AI enthusiasts!"),
    ("Speaker 2", "Hmm, hmm! So, it's like a multi-tool for AI? Can it do backflips? Or solve world hunger?"),
    ("Speaker 1", "Well, it's not exactly solving world hunger yet, but it's making developers super happy by giving them unprecedented control over AI models. It's a significant upgrade from previous versions, with better performance, efficiency, and customization options—kinda like the iPhone of AI models!"),
    ("Speaker 2", "[sigh] That sounds amazing! Wait, does that mean I can teach it how to make me a perfect latte? Or write my memoirs?"),
    ("Speaker 1", "While it's not quite a barista or a ghostwriter yet, it's paving the way for incredible possibilities. With Llama 3.2, developers can create everything from chatbots to recommendation systems, all with a level of customization that was previously out of reach!"),
    ("Speaker 2", "Umm, umm! So, is it like a blank canvas? Or more like a pre-made cake that you can decorate?"),
    ("Speaker 1", "Great analogy, my creative friend! It's more like a blank canvas with some incredible tools. You can start from scratch or build on existing models—either way, you're in control of the masterpiece!"),
    ("Speaker 2", "[laughs] Oh man, this is so cool! So, if I were to try and use this, what would I need to know? Is it like learning a new language? Or just point-and-click magic?"),
    ("Speaker 1", "Well, it's not exactly point-and-click, but it's not rocket surgery either. Developers need a solid understanding of machine learning and some programming skills, but the payoff is huge! Imagine building your own AI assistant that understands your unique needs—this is the start of that kind of future!"),
    ("Speaker 2", "Hmm, hmm! So, are we basically playing God here? Like, creating these little AI minds that can think and learn?"),
    ("Speaker 1", "Ah, but with great power comes great responsibility! While Llama 3.2 opens up amazing possibilities, it also raises important questions about ethics, privacy, and the impact of AI on society. It's a tool, but how we use it is entirely up to us!"),
    ("Speaker 2", "[sigh] Wow, that's heavy. So, is this the start of a utopian future, or are we about to create some kind of AI apocalypse?"),
    ("Speaker 1", "Well, let's not get ahead of ourselves. Llama 3.2 is just one piece of the puzzle, but it's a crucial one. With responsible innovation and a commitment to understanding the implications, we can unlock a future where AI enhances our lives in truly meaningful ways!"),
    ("Speaker 2", "[laughs] Okay, I'm sold! This is way too cool to pass up. Thanks for walking me through it, Captain AI!"),
    ("Speaker 1", "Anytime, my eager friend! And remember, the future is yours to shape—so get out there and start building!"),
    ("Speaker 2", "Umm, umm! Wait, can I have the code now? Or do I need to wait in line?"),
    ("Speaker 1", "The code is already out there, waiting for you to grab it! Go forth and conquer, my digital hero!"),
    ("Speaker 2", "[laughs] Oh, I'm gonna conquer alright! Thanks again!"),
    ("Speaker 1", "You're welcome! And remember, the sky's the limit—unless you're creating AI-powered skywriting, in which case, please keep it PG-13!"),
    ("Speaker 2", "[sigh] Oh man, now I gotta think of something to write in the sky with AI... Thanks a lot!"),
    ("Speaker 1", "Don't mention it! That's what I'm here for!")
    ]
    print("list")
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(lists_with_tuples)
    final_audio = None
    for conv in lists_with_tuples:
        speaker, text = conv[0], conv[1]
        if speaker == "Speaker 1":
            audio_arr, rate = self.generate_audio(text.encode('utf-8', errors='replace').decode('utf-8'),'bm_lewis')
        else:
            audio_arr, rate = self.generate_audio(text.encode('utf-8', errors='replace').decode('utf-8'),'am_michael')
        
        audio_segment = self.numpy_to_audio_segment(audio_arr, rate)
        
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment
    
        
    buffer = io.BytesIO()
    final_audio.export(buffer, format="wav")
    audio_data = buffer.getvalue()
    base64_audio = base64.b64encode(audio_data).decode('utf-8')
    
    return {"generated_podcast":base64_audio}
    # final_audio.export("podcast.mp3", 
    #               format="mp3", 
    #               bitrate="192k",
    #               parameters=["-q:a", "0"])

  def generate_audio(self,text,voice):
    """Generate audio using ParlerTTS for Speaker 1"""
    generator = self.tts_pipeline(
        text, voice=voice, # <= change voice here
        speed=1.2, split_pattern=r'\n+'
    )
    for i, (gs, ps, audio) in enumerate(generator):
        print("working",flush=True)  # i => index
        #print(gs) # gs => graphemes/text
        #print(ps) # ps => phonemes
        # display(Audio(data=audio, rate=24000, autoplay=i==0))
        # sf.write(f'{i}.wav', audio, 24000) # save each audio file

    return audio, 24000
  
  def numpy_to_audio_segment(self,audio_arr, sampling_rate):
    if isinstance(audio_arr, torch.Tensor):
        audio_arr = audio_arr.cpu().numpy()
    
    if audio_arr.dtype != np.float32:
        audio_arr = audio_arr.astype(np.float32)
    
    audio_arr = np.clip(audio_arr, -1, 1)
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    
    return AudioSegment.from_wav(byte_io)
