# Customer Service Processors

## CSChatSplitterProcessor

### Task Summary
The CSChatSplitterProcessor parses customer service chat transcripts and splits them into individual messages with role information. It handles both standard message formats and embedded messages within content.

### Input
- **input_text**: A string containing a customer service chat transcript with role markers like [bot]:, [customer]:, or [agent]:

### Output
- **List[Dict]**: A list of message dictionaries, each containing:
  - **role**: The speaker's role (bot, customer, or agent)
  - **content**: The message content

### Features
- Identifies role markers using regular expressions
- Extracts message content between role markers
- Handles embedded messages within content
- Cleans and normalizes message content
- Preserves the conversation flow and speaker roles

## CSAdapter

### Task Summary
The CSAdapter converts structured chat messages back into a format suitable for dialogue processing, essentially performing the reverse operation of CSChatSplitterProcessor.

### Input
- **chat_messages**: A list of message dictionaries, each containing 'role' and 'content' keys

### Output
- **List[str]**: A list of formatted message strings with role markers

### Features
- Converts structured message dictionaries back to formatted strings
- Preserves role information in the formatted output
- Creates a format compatible with dialogue processing tools

## Example Usage
```python
from src.processing.cs_processor import CSChatSplitterProcessor, CSAdapter

# Sample chat transcript
chat_transcript = """
[customer]: Hello, I need help with my order.
[agent]: Hi there! I'd be happy to help. Could you provide your order number?
[customer]: Yes, it's ABC123456.
[bot]: I've found your order. It's currently being processed.
[customer]: When will it be delivered? [agent]: We expect delivery by Friday.
"""

# Split into structured messages
splitter = CSChatSplitterProcessor()
messages = splitter.process(chat_transcript)

# Example output:
# [
#   {'role': 'customer', 'content': 'Hello, I need help with my order.'},
#   {'role': 'agent', 'content': 'Hi there! I'd be happy to help. Could you provide your order number?'},
#   {'role': 'customer', 'content': 'Yes, it's ABC123456.'},
#   {'role': 'bot', 'content': 'I've found your order. It's currently being processed.'},
#   {'role': 'customer', 'content': 'When will it be delivered?'},
#   {'role': 'agent', 'content': 'We expect delivery by Friday.'}
# ]

# Convert back to formatted strings
adapter = CSAdapter()
formatted_messages = adapter.process(messages)

# Example output:
# [
#   '[customer]: Hello, I need help with my order.',
#   '[agent]: Hi there! I'd be happy to help. Could you provide your order number?',
#   '[customer]: Yes, it's ABC123456.',
#   '[bot]: I've found your order. It's currently being processed.',
#   '[customer]: When will it be delivered?',
#   '[agent]: We expect delivery by Friday.'
# ]
```

## Use Cases
- Processing customer service chat logs
- Preparing conversation data for dialogue models
- Analyzing customer-agent interactions
- Converting between different chat formats
- Extracting structured data from conversation transcripts
