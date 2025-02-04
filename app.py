from utils import extract_list_of_tuples, download_pdf, set_seed, extract_text_from_pdf
import base64
import io
import torch
import transformers
import re
from kokoro import KPipeline
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
import inferless
from pydantic import BaseModel, Field

@inferless.request
class RequestObjects(BaseModel):
  pdf_url: str = Field(default="https://arxiv.org/pdf/2502.01068")

@inferless.response
class ResponseObjects(BaseModel):
  generated_podcast: str = Field(default='Test output')

class InferlessPythonModel:
  def initialize(self):
    set_seed(seed=1526892603)
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
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

  def infer(self, request: RequestObjects) -> ResponseObjects:
    pdf_file = download_pdf(request.pdf_url)
    
    extracted_text = extract_text_from_pdf(pdf_file)
    messages = [
        {"role": "system", "content": self.CREATOR_PROMPT},
        {"role": "user", "content": extracted_text},
    ]

    outputs = self.model_pipeline(
        messages,
        max_new_tokens=8126,
        temperature=1,
    )

    cleaned_content = re.sub(r'<think>.*?</think>', '', outputs[0]["generated_text"][-1]['content'], flags=re.DOTALL)
    
    messages = [
        {"role": "system", "content": self.REFINE_PROMPT},
        {"role": "user", "content": cleaned_content},
    ]
    outputs_refine = self.model_pipeline(
    messages,
    max_new_tokens=8126,
    temperature=1)

    outputs_refine_text = outputs_refine[0]["generated_text"][-1]['content']

    cleaned_outputs_text = re.sub(r'<think>.*?</think>', '', outputs_refine_text, flags=re.DOTALL)
    cleaned_outputs_text = re.sub(r'```python\n|```\n?', '', cleaned_outputs_text)

    lists_with_tuples = extract_list_of_tuples(cleaned_outputs_text)
    
    generated_segments = []
    sampling_rates = []

    final_audio = None
    for conv in lists_with_tuples:
        speaker, text = conv[0], conv[1]
        if speaker == "Speaker 1":
            audio_arr, rate = self.generate_audio(text,'bm_lewis')
        else:
            audio_arr, rate = self.generate_audio(text,'am_michael')
        
        audio_segment = self.numpy_to_audio_segment(audio_arr, rate)
        
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment
    
        
    buffer = io.BytesIO()
    final_audio.export(buffer, format="wav")
    audio_data = buffer.getvalue()
    base64_audio = base64.b64encode(audio_data).decode('utf-8')    
    
    generateObject = ResponseObjects(generated_podcast = base64_audio)        
    return generateObject

  def generate_audio(self,text,voice):
    """Generate audio using ParlerTTS for Speaker 1"""
    generator = self.tts_pipeline(
        text, voice=voice, # <= change voice here
        speed=1.2, split_pattern=r'\n+'
    )

    *_, last_item = generator

    return last_item[2], 24000
  
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
