import re
from bs4 import BeautifulSoup
from typing import List, Union, Dict, Optional

from .processors import Processor


# Processor 1: Text Normalization
class TextNormalizationProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "text_normalization_processor"

    def process(self, input_text: Union[str, List[str]]) -> Union[str, List[str]]:
        def _norm(s: str) -> str:
            s = s.strip().lower()
            return re.sub(r'\s+', ' ', s)

        if isinstance(input_text, list):
            return [_norm(msg) for msg in input_text]
        else:
            return _norm(input_text)

    
# Processor 1: Text Normalization
class TextUpperProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "text_upper_processor"

    def process(self, input_text: str):
        # Basic normalization: trim and lowercase.
        normalized = input_text.strip().upper()
        # Collapse multiple spaces into one.
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

#=====================================================================================
# Processor 2: Dialogue Splitting
class DialogueSplitterProcessor(Processor):
    def __init__(self, min_length: int = 1):
        """
        Args:
            min_length: Minimum number of non-whitespace characters required to keep a message.
        """
        super().__init__()
        self.processor_name = "dialogue_splitter_processor"
        self.min_length = min_length

    def process(self, input_text: str):
        """
        Splits the dialogue into individual messages based on [bom] and [eom] delimiters.
        Returns:
            List of message strings.
        """
        pattern = r'\[bom\](.*?)\[eom\]'
        raw_messages = re.findall(pattern, input_text, flags=re.DOTALL)

        # Strip whitespace and filter out short/empty messages
        messages = [
            msg.strip()
            for msg in raw_messages
            if msg.strip() and len(msg.strip()) >= self.min_length
        ]

        return messages

    
#=====================================================================================
# Processor 3: Dialogue Chunker
class DialogueChunkerProcessor(Processor):
    def __init__(self, 
                 tokenizer, 
                 max_tokens=512, 
                 truncate: bool = False,  # Added truncate parameter
                 max_total_chunks: Optional[int] = 5
                ):
        """
        Args:
            tokenizer: A Hugging Face AutoTokenizer instance.
            max_tokens: Maximum token count per chunk.
        """
        super().__init__()
        self.processor_name = "dialogue_chunker_processor"
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_total_chunks = max_total_chunks
        self.truncate = truncate

    def process(self, messages: List[str]):
        """
        Chunks a list of messages into groups such that each chunk's token count (without special tokens)
        does not exceed the max_tokens limit.
        
        Returns:
            List of dialogue chunks (each chunk is a concatenated string of messages).
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        num_chunks = 0  # Track the number of chunks created

        for msg in messages:
            # Count tokens using the HF AutoTokenizer; avoid adding special tokens here
            token_count = len(self.tokenizer.encode(msg, add_special_tokens=False))
            # If adding this message would exceed limit, save current chunk and start a new one.
            if current_tokens + token_count > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
                    num_chunks += 1 # Increment chunk count
                    if self.max_total_chunks is not None and self.truncate and num_chunks >= self.max_total_chunks:
                        break  # Stop if max_total_chunks is reached
                current_chunk = [msg]
                current_tokens = token_count
            else:
                current_chunk.append(msg)
                current_tokens += token_count

            if self.max_total_chunks is not None and self.truncate and num_chunks >= self.max_total_chunks:
                break # Stop if max_total_chunks is reached

        if current_chunk and (not self.truncate or num_chunks < self.max_total_chunks):
            chunks.append(" ".join(current_chunk).strip())
            num_chunks += 1

        # Ensure at least one non-empty chunk exists
        if not chunks:
            chunks = ["."]
        elif all(not chunk.strip() for chunk in chunks):
            chunks = ["."]

        return chunks
    
    
#====================================================================================
# --- Processor 4: Emoji Remover ---
class EmojiRemoverProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = 'emoji_remover_processor'
        self.emoji_pattern = re.compile(
            "["                                     
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"  # dingbats
            u"\U000024C2-\U0001F251" 
            "]+", 
            flags=re.UNICODE
        )

    def process(self, input_text: Union[str, List[str]]) -> Union[str, List[str]]:
        def _remove(s: str) -> str:
            return self.emoji_pattern.sub('', s)

        if isinstance(input_text, list):
            return [_remove(msg) for msg in input_text]
        else:
            return _remove(input_text)

    
#=====================================================================================   
# --- Processor 2: HTML Normalization ---
class HTMLNormalizerProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "html_normalizer_processor"
    
    def process(self, input_text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        If given a list of dialogue messages, normalize each one; 
        otherwise normalize the single HTML string.
        """
        def _norm_single(text: str) -> str:
            soup = BeautifulSoup(text, "html.parser")
            # collapse whitespace and strip
            return soup.get_text(separator=" ", strip=True)

        if isinstance(input_text, list):
            # apply to each chunk/message
            return [_norm_single(msg) for msg in input_text]
        else:
            return _norm_single(input_text)
           