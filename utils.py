import torch
from pypdf import PdfReader
import requests
import uuid

import re
import ast

def extract_list_of_tuples(content):
    pattern = re.compile(r'\[.*?\]', re.DOTALL)
    candidate_matches = pattern.finditer(content)
    
    valid_lists = []
    
    for match in candidate_matches:
        candidate = match.group(0)
        try:
            # Try to safely evaluate the candidate string as a Python literal.
            parsed_candidate = ast.literal_eval(candidate)
        except Exception:
            # If evaluation fails (e.g., because the string is not a valid literal), skip it.
            continue
        
        # Check that the evaluated object is a list.
        if isinstance(parsed_candidate, list):
            # Optionally, check that every element in the list is a tuple.
            # Adjust the tuple check as needed (e.g., checking tuple length).
            if all(isinstance(item, tuple) for item in parsed_candidate):
                valid_lists.append(parsed_candidate)
    
    return valid_lists


def download_pdf(url):
    try:
        response = requests.get(url, stream=True)  # Stream to handle large files
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        pdf_name = f"{uuid.uuid4()}.pdf"
        with open(pdf_name, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    pdf_file.write(chunk)
        print(f"PDF successfully downloaded: {save_path}")
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

    print(f'Random seed {seed} has been set.')

def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> Optional[str]:
    try:
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PdfReader(file)
            
            
            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")
            
            extracted_text = []
            total_chars = 0
            
            # Iterate through all pages
            for page_num in range(num_pages):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Check if adding this page's text would exceed the limit
                if total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break
                
                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")
            
            final_text = '\n'.join(extracted_text)
            print(f"\nExtraction complete! Total characters: {len(final_text)}")
            return final_text
            
    except PyPDF2.PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None