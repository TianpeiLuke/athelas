# BSM Processors

This document describes the various text processing components used in the Buyer-Seller Messaging (BSM) system.

## TextNormalizationProcessor

### Task Summary
Normalizes text by converting to lowercase, trimming whitespace, and collapsing multiple spaces into a single space.

### Input
- **input_text**: A string or list of strings containing the text to normalize

### Output
- A normalized string or list of normalized strings with consistent spacing and lowercase formatting

## TextUpperProcessor

### Task Summary
Normalizes text by converting to uppercase, trimming whitespace, and collapsing multiple spaces into a single space.

### Input
- **input_text**: A string containing the text to convert to uppercase

### Output
- A normalized string with consistent spacing and uppercase formatting

## DialogueSplitterProcessor

### Task Summary
Splits dialogue text into individual messages based on [bom] and [eom] delimiters, filtering out empty or short messages.

### Input
- **input_text**: A string containing dialogue with [bom] and [eom] markers
- **min_length**: Minimum number of non-whitespace characters required to keep a message

### Output
- A list of individual message strings extracted from the dialogue

## DialogueChunkerProcessor

### Task Summary
Chunks a list of messages into groups such that each chunk's token count does not exceed a specified limit, useful for processing with models that have token limits.

### Input
- **messages**: A list of message strings
- **tokenizer**: A Hugging Face AutoTokenizer instance
- **max_tokens**: Maximum token count per chunk
- **truncate**: Whether to truncate the output if max_total_chunks is reached
- **max_total_chunks**: Maximum number of chunks to return

### Output
- A list of dialogue chunks, where each chunk is a concatenated string of messages that fits within the token limit

## EmojiRemoverProcessor

### Task Summary
Removes emoji characters from text using Unicode pattern matching.

### Input
- **input_text**: A string or list of strings containing text with potential emoji characters

### Output
- The input text with all emoji characters removed

## HTMLNormalizerProcessor

### Task Summary
Normalizes HTML content by extracting plain text and removing HTML tags and formatting.

### Input
- **input_text**: A string or list of strings containing HTML content

### Output
- Plain text extracted from the HTML content with consistent spacing
