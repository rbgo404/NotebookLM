import torch
from pypdf import PdfReader
import requests
import uuid
import random
import numpy as np
import re
import ast

def extract_list_of_tuples(content):
    pattern = re.compile(r'\[\s*((?:\s*\("Speaker \d+",\s*"[^"]+"\),?\s*)+)\s*\]', re.DOTALL)
    candidate_matches = pattern.finditer(content)
    
    valid_lists = []
    for match in candidate_matches:
        candidate = match.group(0)
        try:
            parsed_candidate = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(parsed_candidate, list):
            if all(isinstance(item, tuple) for item in parsed_candidate):
                return parsed_candidate

def download_pdf(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        pdf_name = f"{uuid.uuid4()}.pdf"
        with open(pdf_name, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    pdf_file.write(chunk)
        print(f"PDF successfully downloaded")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF: {e}")
    
    return pdf_name
    
def set_seed(seed=None, seed_torch=True):
    if seed is None:
      seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.cuda.manual_seed(seed)
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True

def extract_text_from_pdf(file_path: str, max_chars: int = 100000):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            num_pages = len(pdf_reader.pages)
            extracted_text = []
            total_chars = 0
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if total_chars + len(text) > max_chars:
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    break
                
                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")
            
            final_text = '\n'.join(extracted_text)
            return final_text
            
    except PyPDF2.PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None