import re
from typing import List, Union, Optional, Dict

from .processors import Processor


class CSChatSplitterProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "cs_chat_splitter_processor"

    def process(self, input_text: str) -> List[Dict[str, str]]:
        """Split chat into individual messages with roles."""
        messages = []
        
        # Use regex to find all role markers and their positions
        pattern = r'\[(bot|customer|agent)\]:'
        markers = list(re.finditer(pattern, input_text))
        
        # Process each message segment
        for i in range(len(markers)):
            current_match = markers[i]
            next_match = markers[i + 1] if i < len(markers) - 1 else None
            
            # Get current message boundaries
            start_pos = current_match.end()
            end_pos = next_match.start() if next_match else len(input_text)
            
            # Extract role and content
            role = current_match.group(1)
            content = input_text[start_pos:end_pos].strip()
            
            # Check for embedded messages in content
            embedded_messages = self._extract_embedded_messages(content)
            
            if embedded_messages:
                # Add the main message (content before first embedded)
                main_content = content[:content.find('[')].strip()
                if main_content:
                    messages.append({
                        'role': role,
                        'content': main_content
                    })
                # Add all embedded messages
                messages.extend(embedded_messages)
            else:
                if content:
                    messages.append({
                        'role': role,
                        'content': content
                    })

        return messages

    def _extract_embedded_messages(self, content: str) -> List[Dict[str, str]]:
        """Extract any embedded messages from content."""
        embedded = []
        pattern = r'\[(bot|customer|agent)\]:([^[]+)(?=\[|$)'
        
        matches = re.finditer(pattern, content)
        for match in matches:
            role = match.group(1)
            message_content = match.group(2).strip()
            if message_content:
                embedded.append({
                    'role': role,
                    'content': message_content
                })
        
        return embedded

    def _clean_content(self, content: str) -> str:
        """Clean message content."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove any remaining role markers
        content = re.sub(r'\[(bot|customer|agent)\]:', '', content)
        
        return content.strip()

    
    
    
class CSAdapter(Processor):
    def __init__(self):
        self.processor_name = "cs_adapter_processor"

    def process(self, chat_messages: List[Dict[str, str]]) -> List[str]:
        """
        Converts the output of ChatSplitterProcessor to a format suitable for DialogueChunkerProcessor.

        Args:
            chat_messages (List[Dict[str, str]]): List of dictionaries, each containing 'role' and 'content' keys.

        Returns:
            List[str]: List of formatted message strings.
        """
        formatted_messages = []

        for message in chat_messages:
            role = message['role']
            content = message['content']
            formatted_message = f"[{role}]: {content}"
            formatted_messages.append(formatted_message)

        return formatted_messages