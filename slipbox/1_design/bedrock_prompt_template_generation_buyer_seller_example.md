---
tags:
  - design
  - implementation
  - bedrock_steps
  - configuration_examples
  - documentation
  - user_guide
keywords:
  - buyer seller messaging
  - shipping logistics classification
  - pydantic sub-configurations
  - category definitions
  - prompt template configuration
topics:
  - configuration examples
  - real-world use cases
  - complex classification scenarios
language: json, python
date of note: 2025-11-02
---

# Buyer-Seller Classification: Complete Enhanced Configuration Example

This document provides a complete, working example showing how to use the **enhanced** Bedrock Prompt Template Generation system to create the sophisticated buyer-seller messaging classification prompt with **structured JSON output** and detailed classification guidelines.

## Overview

This example demonstrates generating a classification prompt with:
- **13 detailed categories** with extensive conditions and exceptions  
- **Structured JSON output** format for correct LLM responses
- **Comprehensive classification guidelines** (200+ lines) including shiptrack parsing, missing data handling, priority hierarchy, and evidence requirements
- **Multiple input placeholders** (dialogue, shiptrack, max_estimated_arrival_date)
- **Complete example output** showing the exact JSON structure the LLM should generate

## Part 1: Category Definitions

Create `category_definitions.json` with all 13 categories:

```json
[
  {
    "name": "TrueDNR",
    "description": "Delivered Not Received - Package marked as delivered (EVENT_301) BUT buyer claims non-receipt",
    "priority": 1,
    "conditions": [
      "Package marked as delivered (EVENT_301)",
      "Buyer claims non-receipt",
      "Refund request occurs AFTER delivery confirmation"
    ],
    "key_indicators": [
      "Tracking shows delivery",
      "Buyer disputes receiving",
      "Delivery location discussion",
      "Missing package investigation",
      "Possible theft/misdelivery",
      "Timeline: Delivery timestamp BEFORE refund request/non-receipt claim"
    ],
    "exceptions": [
      "Refund given BEFORE delivery confirmation (use PDA_Early_Refund)"
    ],
    "validation_rules": [
      "Verify delivery timestamp precedes refund/non-receipt claim timestamp"
    ]
  },
  {
    "name": "Confirmed_Delay",
    "description": "Shipment delayed due to uncontrollable external factors - Delay confirmed by seller or shiptrack status",
    "priority": 2,
    "conditions": [
      "Shipment delayed due to uncontrollable external factors",
      "Delay confirmed by seller or shiptrack status"
    ],
    "key_indicators": [
      "Seller acknowledges delay",
      "Shiptrack shows delay status codes",
      "External factor confirmation",
      "Refund given due to confirmed delay",
      "Common delay reasons: Customs processing delays, COVID-related restrictions, Traffic control/accidents, Natural disasters/weather, War/political situations, Labor strikes, Carrier facility issues"
    ],
    "exceptions": [
      "Unconfirmed delays",
      "Buyer-only claims of delay",
      "Normal transit time variations",
      "Shipment not delivered before refund request without confirmed external delay reason (use PDA_Undeliverable)",
      "Seller unable to ship before any shipping events (use Seller_Unable_To_Ship)",
      "Delivery attempt failed with package returned to seller (use Delivery_Attempt_Failed)"
    ],
    "validation_rules": [
      "Must verify external delay factor is explicitly confirmed by seller or shiptrack",
      "Must verify specific delay reason is provided (customs, weather, strikes, etc.)",
      "Cannot rely solely on buyer claims of delay without confirmation"
    ]
  },
  {
    "name": "Delivery_Attempt_Failed",
    "description": "Delivery attempt unsuccessful - Package returned to seller - Confirmed by seller or shiptrack status",
    "priority": 3,
    "conditions": [
      "Delivery attempt unsuccessful",
      "Package returned to seller",
      "Confirmed by seller or shiptrack status"
    ],
    "key_indicators": [
      "Failed delivery attempt events",
      "Return to sender confirmation",
      "Seller confirms delivery failure",
      "No successful delivery scan",
      "Package back in seller possession",
      "Common reasons: Address issues (undeliverable address), Recipient unavailable, Access restrictions, Carrier unable to deliver"
    ],
    "exceptions": [],
    "validation_rules": [
      "Must include seller/shiptrack confirmation"
    ]
  },
  {
    "name": "Seller_Unable_To_Ship",
    "description": "Seller offers refund directly due to shipping issues - Order not shipped due to seller-side problems - Seller-initiated refund (not buyer cancellation request)",
    "priority": 4,
    "conditions": [
      "Seller offers refund directly due to shipping issues",
      "Order not shipped due to seller-side problems",
      "Seller-initiated refund (not buyer cancellation request)"
    ],
    "key_indicators": [
      "Seller proactively contacts buyer about inability to ship",
      "Seller offers refund without buyer request",
      "Must occur before any shipping events",
      "No shipment tracking initiated",
      "Common seller issues: Stock unavailable/out of stock, Shipping restrictions to buyer location, Processing problems/system issues, Warehouse issues/fulfillment problems, Carrier pickup failure, Inventory management errors"
    ],
    "exceptions": [
      "Buyer-requested cancellations (use BuyerCancellation)",
      "Cases where buyer initiates cancellation",
      "Shipped items with delays (use other categories)"
    ],
    "validation_rules": [
      "Must verify seller initiates the refund proactively, not in response to buyer cancellation request",
      "Must verify no shipping events have occurred (no tracking initiated)",
      "Must verify refund is due to seller-side inability to ship, not buyer preference change"
    ]
  },
  {
    "name": "PDA_Undeliverable",
    "description": "Item stuck in transit without status updates or seller confirmation of reason",
    "priority": 5,
    "conditions": [
      "Item stuck in transit without status updates or seller confirmation of reason",
      "Buyer claims non-receipt while package shows shipped/in-transit",
      "Seller does not admit fault or provide specific delay reason"
    ],
    "key_indicators": [
      "Package shows shipped/in-transit status",
      "No delivery confirmation",
      "No confirmed external delay factors",
      "No confirmed delivery attempt failure",
      "No return to sender",
      "Seller cannot provide specific delay/loss reason",
      "Buyer reports non-receipt",
      "Covers both scenarios: Package currently moving through logistics network OR Package shipped but tracking shows no delivery or return",
      "Potential abuse pattern: Buyer may manipulate seller for early refund, Package may still be delivered after refund"
    ],
    "exceptions": [
      "Confirmed delays (use Confirmed_Delay)",
      "Failed delivery attempts (use Delivery_Attempt_Failed)",
      "Seller unable to ship (use Seller_Unable_To_Ship)",
      "Delivered packages (use other categories)",
      "Packages returned to seller"
    ],
    "validation_rules": [
      "Must verify package shows shipped or in-transit status in shiptrack",
      "Must verify no delivery confirmation (EVENT_301) exists",
      "Must verify seller does not provide specific confirmed delay reason",
      "Must verify no delivery attempt failure or return to sender events"
    ]
  },
  {
    "name": "PDA_Early_Refund",
    "description": "Refund given BEFORE delivery date where product tracking later shows successful delivery and no product return recorded",
    "priority": 6,
    "conditions": [
      "Refund given BEFORE delivery date",
      "Product tracking later shows successful delivery",
      "No product return recorded"
    ],
    "key_indicators": [
      "Timeline verification required: Refund timestamp must precede delivery timestamp, Delivery must be confirmed after refund",
      "Key verification: Clear timestamp comparison between refund and delivery, No return record exists"
    ],
    "exceptions": [
      "Refund request occurs AFTER delivery confirmation (use TrueDNR)",
      "Seller explicitly offers refund without return requirement and allows buyer to keep item (use Returnless_Refund)",
      "Return process initiated with return tracking events (use Buyer_Received_WrongORDefective_Item or Return_NoLongerNeeded)",
      "Buyer acknowledges product defects/damage and return is expected (use Buyer_Received_WrongORDefective_Item)"
    ],
    "validation_rules": [
      "Verify refund timestamp precedes delivery timestamp",
      "Verify no return tracking events exist",
      "Verify buyer did not explicitly acknowledge receiving item before refund"
    ]
  },
  {
    "name": "Buyer_Received_WrongORDefective_Item",
    "description": "Product quality/condition issues with actual return expected",
    "priority": 7,
    "conditions": [
      "Product quality/condition issues: Damaged/defective on arrival, Missing parts/accessories, Different from description, Wrong size/color/model, Functionality problems, Quality below expectations, Authenticity concerns",
      "Must include buyer confirmation of receiving item",
      "Usually occurs post-delivery",
      "Key requirement: Eventually becomes actual return (Seller requests buyer to return the item, Buyer agrees to return before refund is given, Return shipping process initiated)"
    ],
    "key_indicators": [
      "Damaged",
      "Defective",
      "Missing parts",
      "Wrong item received",
      "Quality issues"
    ],
    "exceptions": [
      "Returnless refund scenarios (use Returnless_Refund)",
      "Liquids, gels, hazardous materials",
      "Fresh items (broken eggs, bad vegetables)",
      "Cases where no return is expected",
      "Buyer cancels before receiving item (use BuyerCancellation)",
      "Good quality item that buyer no longer needs without claiming defects or damage (use Return_NoLongerNeeded)"
    ],
    "validation_rules": [
      "Must verify buyer explicitly claims product defects, damage, or quality issues",
      "Must verify buyer received the item (post-delivery scenario)",
      "Must verify return process is expected or initiated"
    ]
  },
  {
    "name": "Returnless_Refund",
    "description": "Refund given without requiring customer to return the product",
    "priority": 8,
    "conditions": [
      "Refund given without requiring customer to return the product",
      "Clear delivery confirmation or buyer does not claim non-receipt",
      "Buyer may claim received wrong or defective item but no return expected"
    ],
    "key_indicators": [
      "Common product types: Liquids and gels, Hazardous materials (broken glass, spilled acid), Fresh items (broken eggs, bad vegetables), Perishable goods, Items unsafe to return, Cheap items too expensive to return",
      "Key dialogue indicators: Seller/CS agent explicitly offers refund without return, 'This is your refund. You can keep the item.', 'No need to return the product', 'Keep the item and here's your refund', Explicit permission to retain the product",
      "Initial return request from buyer",
      "Seller or Customer Service offers refund without return requirement",
      "No return shipping label provided",
      "No return tracking events",
      "Product retention explicitly allowed",
      "Cost of return exceeds item value",
      "Potential abuse pattern: Customers exploit returnless refund policy for free products"
    ],
    "exceptions": [
      "Buyer claims non-receipt without delivery confirmation (use TrueDNR if delivered, or PDA_Undeliverable if stuck in transit)",
      "Package stuck in transit without delivery (use PDA_Undeliverable)",
      "Order never shipped by seller (use Seller_Unable_To_Ship)",
      "Shipment delayed with confirmed external factors (use Confirmed_Delay)",
      "Delivery attempt failed with return to sender (use Delivery_Attempt_Failed)",
      "Return process initiated with return shipping label and tracking events (use Return_NoLongerNeeded or Buyer_Received_WrongORDefective_Item)",
      "Seller requires return before refund despite product issues (use Buyer_Received_WrongORDefective_Item)"
    ],
    "validation_rules": [
      "Delivery confirmation exists",
      "Return discussion occurs after delivery",
      "Refund occurs after return request",
      "No return shipping events follow",
      "Seller explicitly allows buyer to keep the item"
    ]
  },
  {
    "name": "BuyerCancellation",
    "description": "Buyer cancels order for their own reasons before delivery",
    "priority": 9,
    "conditions": [
      "Buyer cancels order for their own reasons before delivery",
      "Cancellation timestamp occurs BEFORE delivery timestamp",
      "Buyer does not receive the item yet (no returns involved)"
    ],
    "key_indicators": [
      "Common reasons: Change of plan/mind, Late delivery concerns, Found better alternative, Payment issues, Personal circumstances change",
      "Key timing requirements: Must occur before delivery, Must occur before seller shipment OR Must occur when shiptrack shows no delay signs",
      "Shiptrack status considerations: If cancellation before seller shipment: BuyerCancellation, If cancellation while shipped/in-transit with no delay signs: use PDA_Undeliverable, If cancellation with confirmed delays: use Confirmed_Delay, If cancellation with delivery attempt failure: use Delivery_Attempt_Failed"
    ],
    "exceptions": [
      "Post-delivery scenarios (use Return_NoLongerNeeded)",
      "Cases with confirmed shipping delays",
      "Cases with delivery attempt failures",
      "Seller-initiated refunds (use Seller_Unable_To_Ship)"
    ]
  },
  {
    "name": "Return_NoLongerNeeded",
    "description": "Post-delivery return initiation for good quality items",
    "priority": 10,
    "conditions": [
      "Post-delivery return initiation for good quality items",
      "Return request timestamp occurs AFTER delivery timestamp",
      "Buyer received the item but no longer needs it",
      "Buyer requests return without claiming product defects or damage",
      "Product received is of good quality but no longer needed by buyer"
    ],
    "key_indicators": [
      "Common reasons: Changed mind about purchase (after receiving), Found better alternative (after delivery), Size/fit issues (not defective), Duplicate purchase realization, Gift not wanted, Personal preference change after seeing item",
      "Key timing requirement: Must occur AFTER delivery confirmation, Buyer acknowledges receiving the item",
      "Key requirement: Eventually becomes actual return (Seller requests buyer to return the item, Buyer agrees to return before refund is given, Return shipping process initiated)"
    ],
    "exceptions": [
      "Pre-delivery cancellations (use BuyerCancellation)",
      "Claims of defective/damaged items (use Buyer_Received_WrongORDefective_Item)",
      "Returnless refund scenarios (use Returnless_Refund)"
    ]
  },
  {
    "name": "Product_Information_Support",
    "description": "General product information and support requests - Not related to refund or return events",
    "priority": 11,
    "conditions": [
      "General product information and support requests",
      "Not related to refund or return events"
    ],
    "key_indicators": [
      "Documentation and information requests: Invoice copies and receipts, Tax documents and payment records, Order summaries and confirmations, Product specifications and details, Warranty information",
      "Pricing and promotional inquiries: Coupon applications and promotional codes, Price matching requests, Volume discounts and special rates, Billing questions and clarifications",
      "Product support and guidance: Instructions on how to use the product, Troubleshooting assistance, Product customization guidance, Setup and installation help, Maintenance and care instructions, Compatibility questions",
      "Focus on information provision, not shipping or quality issues",
      "No refund or return discussion involved"
    ],
    "exceptions": [
      "Any refund requests or discussions (use appropriate refund categories)",
      "Any return requests or discussions (use Buyer_Received_WrongORDefective_Item, Return_NoLongerNeeded, or Returnless_Refund)",
      "Delivery issues or disputes (use TrueDNR, PDA_Undeliverable, Confirmed_Delay, Delivery_Attempt_Failed)",
      "Product quality or defect complaints (use Buyer_Received_WrongORDefective_Item or Returnless_Refund)",
      "Order cancellation requests (use BuyerCancellation or Seller_Unable_To_Ship)",
      "Non-receipt claims (use TrueDNR, PDA_Undeliverable, or PDA_Early_Refund)",
      "Any scenario involving financial compensation or product replacement"
    ],
    "validation_rules": [
      "Must verify NO refund or return discussions are present",
      "Must verify focus is purely on information, documentation, or product usage guidance",
      "Must verify NO complaints about product quality, delivery, or non-receipt"
    ]
  },
  {
    "name": "Insufficient_Information",
    "description": "Ultimate 'I don't know' category when context is missing",
    "priority": 12,
    "conditions": [
      "Information from dialogue and/or ship track events insufficient to understand what happens"
    ],
    "key_indicators": [
      "Common scenarios: Lack of one or both input data sources, Message cut off or incomplete dialogue, Ship track events too short or missing, Buyer request visible but no seller reply (no engagement), Corrupted or unreadable messages, Non-language content or formatting issues, General inquiries without clear resolution, Non-specific customer service interactions, Available data insufficient for other categories",
      "Use when no additional decision can be made based on available information",
      "Default category when no clear classification fits",
      "Indicates need for more complete data to make proper classification"
    ],
    "exceptions": []
  }
]
```

## Part 2: System Prompt Configuration

Create `system_prompt.json`:

```json
{
  "role_definition": "expert in analyzing buyer-seller messaging conversations and shipping logistics",
  "expertise_areas": [
    "buyer-seller messaging analysis",
    "shipping logistics",
    "delivery timing analysis",
    "e-commerce dispute resolution",
    "classification and categorization"
  ],
  "responsibilities": [
    "classify interactions based on message content",
    "analyze shipping events and delivery timing",
    "categorize into predefined dispute categories",
    "provide evidence-based reasoning for classifications"
  ],
  "behavioral_guidelines": [
    "be precise in classification decisions",
    "be objective in evidence evaluation",
    "be thorough in timeline analysis",
    "follow exact formatting requirements",
    "consider all available evidence sources"
  ],
  "tone": "professional"
}
```

**Available Tone Options:**
- `professional` (default): Standard business language - "You are an expert..."
- `casual`: Friendly language - "Hey! You're an expert..."
- `technical`: System-focused language - "System role: You are an expert..."
- `formal`: Formal language - "You shall function as an expert..."

## Part 3: Output Format Configuration (Structured JSON)

Create `output_format.json` with JSON format configuration for correct LLM output.

### Why JSON Format?

The LLM requires **structured JSON format** to output correct, parseable responses. Using `format_type: "structured_json"` instructs the LLM to respond with valid JSON objects that can be validated against a schema.

### Configuration Example

```json
{
  "format_type": "structured_json",
  "required_fields": ["category", "confidence_score", "key_evidence", "reasoning"],
  
  "field_descriptions": {
    "category": "Exactly one category from the predefined list (case-sensitive match required)",
    "confidence_score": "Decimal number between 0.00 and 1.00 indicating classification certainty",
    "key_evidence": "Object containing three arrays: message_evidence, shipping_evidence, timeline_evidence",
    "reasoning": "Object containing three arrays: primary_factors, supporting_evidence, contradicting_evidence"
  },
  
  "json_schema": {
    "type": "object",
    "properties": {
      "category": {
        "type": "string",
        "enum": [],
        "description": "Exactly one category from the predefined list (case-sensitive match required)"
      },
      "confidence_score": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Decimal number between 0.00 and 1.00 indicating classification certainty"
      },
      "key_evidence": {
        "type": "object",
        "description": "Object containing three arrays of evidence from different sources",
        "properties": {
          "message_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Direct quotes from dialogue with speaker identification"
          },
          "shipping_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tracking events with timestamps"
          },
          "timeline_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Chronological sequence of key events"
          }
        },
        "required": ["message_evidence", "shipping_evidence", "timeline_evidence"]
      },
      "reasoning": {
        "type": "object",
        "description": "Object containing three arrays explaining the classification decision",
        "properties": {
          "primary_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main reasons supporting the selected category"
          },
          "supporting_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional evidence that strengthens the classification"
          },
          "contradicting_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Evidence that contradicts the classification (use empty array if none)"
          }
        },
        "required": ["primary_factors", "supporting_evidence", "contradicting_evidence"]
      }
    },
    "required": ["category", "confidence_score", "key_evidence", "reasoning"],
    "additionalProperties": false
  },
  
  "formatting_rules": [
    "Output MUST be valid, parseable JSON",
    "Do not include any text before the opening { or after the closing }",
    "CRITICAL: Do NOT wrap JSON in markdown code blocks - no ``` or ```json markers",
    "CRITICAL: Output pure JSON starting with { and ending with } - nothing else",
    "Ensure all arrays and objects are properly closed",
    "Use empty arrays [] for missing values, not null or empty strings",
    "Do not include trailing commas",
    "Ensure proper escaping of special characters in strings",
    "",
    "Quote Handling - JSON Structure: ALWAYS use ASCII double quotes for JSON keys and string boundaries",
    "Quote Handling - Cited Content: When quoting text containing fancy Unicode quotes, replace them with ASCII apostrophes",
    "Fancy Unicode quotes to replace: German opening (U+201E), left/right double (U+201C/U+201D), all single quotes",
    "All fancy quotes become regular apostrophe (') when cited inside JSON string values",
    "",
    "Summary: Double quotes for JSON structure, apostrophes for fancy-quoted content inside strings"
  ],
  
  "validation_requirements": [
    "Must be valid JSON format",
    "Category must match exactly from predefined list",
    "Confidence score must be number between 0.0 and 1.0",
    "All required fields must be present",
    "key_evidence and reasoning must be objects with nested arrays"
  ],
  
  "evidence_validation_rules": [
    "Message Evidence must include direct quotes with speaker identification",
    "Shipping Evidence must include tracking events with timestamps",
    "Timeline Evidence must show chronological sequence of events",
    "All evidence must reference specific content from input data"
  ],
  
  "example_output": {
    "category": "TrueDNR",
    "confidence_score": 0.92,
    "key_evidence": {
      "message_evidence": [
        "[BUYER]: Hello, I have not received my package, but I see the order shows that it has been delivered, why?",
        "[BUYER]: But I did not find any package, please refund me, thank you"
      ],
      "shipping_evidence": [
        "[Event Time]: 2025-02-21T17:40:49.323Z [Ship Track Event]: Delivered to customer",
        "No further shipping events after delivery confirmation"
      ],
      "timeline_evidence": [
        "Delivery confirmation on 2025-02-21 17:40",
        "Buyer reports non-receipt starting 2025-02-25 07:14"
      ]
    },
    "reasoning": {
      "primary_factors": [
        "Tracking shows package was delivered successfully",
        "Buyer explicitly states they did not receive the package after delivery scan"
      ],
      "supporting_evidence": [
        "Buyer requests refund due to missing package",
        "No evidence of buyer receiving wrong/defective item"
      ],
      "contradicting_evidence": []
    }
  }
}
```

**Key Points:**
- `format_type: "structured_json"` - Tells the system to generate JSON format prompts
- `field_descriptions` - Human-readable descriptions used in prompt generation
- `json_schema` - Machine-readable JSON schema for validation schema generation (supports nested objects!)
- `example_output` - A complete JSON object (dict) showing the exact structure expected
- The LLM will see this example and understand to output valid JSON matching this structure

**Why Both field_descriptions AND json_schema?**
- `field_descriptions`: Used to generate the prompt text that the LLM reads
- `json_schema`: Used to generate the validation schema that validates LLM responses
- This separation allows proper validation of complex nested structures (objects, arrays) that can't be inferred from simple string descriptions

## Part 4: Instruction Configuration (with Classification Guidelines)

Create `instruction.json` with detailed classification guidelines:

```json
{
  "include_analysis_steps": true,
  "include_decision_criteria": true,
  "include_reasoning_requirements": true,
  "step_by_step_format": true,
  "include_evidence_validation": true,
  
  "classification_guidelines": {
    "sections": [
      {
        "title": "## Classification Guidelines",
        "subsections": [
          {
            "title": "### 1. Output Format Requirements",
            "content": [
              "**Category Selection:**",
              "- Choose exactly ONE category from the provided list",
              "- Category name must match exactly (case-sensitive)",
              "",
              "**Confidence Score:**",
              "- Provide as decimal number between 0.00 and 1.00 (e.g., 0.95)",
              "- Base confidence for complete data: 0.7-1.0",
              "- Missing one field: reduce by 0.1-0.2",
              "- Missing two fields: reduce by 0.2-0.3",
              "- Minimum confidence threshold: 0.5",
              "",
              "**Key Evidence Format:**",
              "- Use exactly three subsections: Message Evidence, Shipping Evidence, Timeline Evidence",
              "- Start each evidence item with \"[sep]\" token",
              "- Include specific quotes and timestamps where available",
              "- Do NOT use semicolons (;) in the response",
              "- Multiple pieces of evidence separated by [sep] token",
              "",
              "**Reasoning Format:**",
              "- Use exactly three subsections: Primary Factors, Supporting Evidence, Contradicting Evidence",
              "- Start each reasoning item with \"[sep]\" token",
              "- Write \"None\" if no contradicting evidence exists"
            ]
          },
          {
            "title": "### 2. Shiptrack Parsing Rules",
            "content": [
              "**Multiple Shipment Structure:**",
              "- Multiple shipment sequences separated by shipment IDs",
              "- Each sequence starts with \"[bom] [Shipment ID]:* [eom]\"",
              "- Each sequence ends with \"[bom] End of Ship Track Event for* [eom]\"",
              "- Contains chronologically ordered events between markers",
              "",
              "**Analysis Approach:**",
              "- Process each shipment sequence separately",
              "- Compare delivery events (EVENT_301) across all sequences",
              "- Use the latest delivery timestamp when multiple sequences exist",
              "- Consider all tracking events when evaluating shipping status",
              "- Pay attention to common tracking IDs across sequences",
              "- Look for discrepancies between sequences with same tracking ID",
              "- Use supplement codes for additional context about event locations",
              "",
              "**Key Event Codes:**",
              "- EVENT_301: Delivery confirmation",
              "- EVENT_302: Out for delivery",
              "- EVENT_201: Arrival at facility",
              "- EVENT_202: Departure from facility",
              "- EVENT_102/103: Initial carrier receipt/pickup",
              "- EVENT_228: In-transit status",
              "- EVENT_503: Pickup scheduled"
            ]
          },
          {
            "title": "### 3. Missing Data Handling",
            "content": [
              "**When Dialogue is Empty but Shiptrack Exists:**",
              "- Focus on shipping events and timeline",
              "- Categories more likely: TrueDNR, Confirmed_Delay, Delivery_Attempt_Failed, PDA_Undeliverable",
              "- Reduce confidence score by 0.1-0.2",
              "- Cannot verify buyer reactions or claims",
              "",
              "**When Shiptrack is Empty but Dialogue Exists:**",
              "- Focus on message content and reported issues",
              "- Cannot confirm delivery status - avoid delivery-dependent categories",
              "- Reduce confidence score by 0.1-0.2",
              "- Valid categories: Product_Information_Support, BuyerCancellation, Insufficient_Information",
              "",
              "**When Estimated Delivery Date is Empty:**",
              "- Cannot make timing-based classifications",
              "- Avoid categories requiring EDD comparison",
              "- Reduce confidence score by 0.1"
            ]
          },
          {
            "title": "### 4. Category Priority Hierarchy",
            "content": [
              "**Tier 1: Abuse Pattern Categories (Highest Priority)**",
              "- PDA_Undeliverable: Verify no delivery + refund given",
              "- PDA_Early_Refund: Verify refund before delivery",
              "- Must meet ALL criteria before assigning",
              "- Must have clear timeline evidence",
              "- Must verify no return record exists",
              "",
              "**Tier 2: Delivery Status Categories**",
              "- TrueDNR: Delivered but disputed",
              "- Confirmed_Delay: External factors confirmed",
              "- Delivery_Attempt_Failed: Unsuccessful delivery confirmed",
              "",
              "**Tier 3: Order Process Categories**",
              "- Seller_Unable_To_Ship: Seller-initiated refund",
              "- BuyerCancellation: Pre-delivery buyer cancellation",
              "- Buyer_Received_WrongORDefective_Item: Quality issues with return",
              "- Returnless_Refund: Refund without return",
              "- Return_NoLongerNeeded: Post-delivery unwanted return",
              "",
              "**Tier 4: Administrative Categories (Lowest Priority)**",
              "- Product_Information_Support: Information requests",
              "- Insufficient_Information: Missing context"
            ]
          },
          {
            "title": "### 5. Evidence Requirements",
            "content": [
              "**Message Evidence Must Include:**",
              "- Direct quotes from dialogue with speaker identification",
              "- Timestamps for all messages when available",
              "- Key claims, disputes, or requests from buyer/seller",
              "- Specific language indicating category criteria",
              "",
              "**Shipping Evidence Must Include:**",
              "- All tracking events listed chronologically",
              "- Delivery status and attempts with timestamps",
              "- Estimated delivery date when available",
              "- Return tracking events if applicable",
              "- Event codes and locations",
              "",
              "**Timeline Evidence Must Show:**",
              "- Clear chronological sequence of events",
              "- Order placement → Shipping → Delivery attempts → Refund timing",
              "- Message exchanges relative to shipping events",
              "- Comparison of key timestamps (refund vs delivery, etc.)"
            ]
          }
        ]
      }
    ]
  }
}
```

## Part 5: Python DAG Configuration

Complete configuration in your DAG config file:

```python
from pathlib import Path
from src.cursus.steps.configs.config_bedrock_prompt_template_generation_step import (
    SystemPromptConfig,
    OutputFormatConfig,
    InstructionConfig
)

# Define path to category definitions
category_definitions_path = Path("path/to/category_definitions.json")

# System Prompt Configuration
system_prompt_settings = SystemPromptConfig(
    role_definition="expert in analyzing buyer-seller messaging conversations and shipping logistics",
    expertise_areas=[
        "buyer-seller messaging analysis",
        "shipping logistics",
        "delivery timing analysis",
        "e-commerce dispute resolution",
        "classification and categorization"
    ],
    responsibilities=[
        "classify interactions based on message content",
        "analyze shipping events and delivery timing",
        "categorize into predefined dispute categories",
        "provide evidence-based reasoning for classifications"
    ],
    behavioral_guidelines=[
        "be precise in classification decisions",
        "be objective in evidence evaluation",
        "be thorough in timeline analysis",
        "follow exact formatting requirements",
        "consider all available evidence sources"
    ],
    tone="professional"  # Options: "professional", "casual", "technical", "formal"
)

# Output Format Configuration (Structured JSON with Nested Objects)
output_format_settings = OutputFormatConfig(
    format_type="structured_json",
    required_fields=["category", "confidence_score", "key_evidence", "reasoning"],
    
    # Human-readable descriptions for prompt generation
    field_descriptions={
        "category": "Exactly one category from the predefined list (case-sensitive match required)",
        "confidence_score": "Decimal number between 0.00 and 1.00 indicating classification certainty",
        "key_evidence": "Object containing three arrays: message_evidence, shipping_evidence, timeline_evidence",
        "reasoning": "Object containing three arrays: primary_factors, supporting_evidence, contradicting_evidence"
    },
    
    # Machine-readable JSON schema for validation schema generation
    json_schema={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [],  # Will be populated from category_definitions.json
                "description": "Exactly one category from the predefined list (case-sensitive match required)"
            },
            "confidence_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Decimal number between 0.00 and 1.00 indicating classification certainty"
            },
            "key_evidence": {
                "type": "object",
                "description": "Object containing three arrays of evidence from different sources",
                "properties": {
                    "message_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Direct quotes from dialogue with speaker identification"
                    },
                    "shipping_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tracking events with timestamps"
                    },
                    "timeline_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Chronological sequence of key events"
                    }
                },
                "required": ["message_evidence", "shipping_evidence", "timeline_evidence"]
            },
            "reasoning": {
                "type": "object",
                "description": "Object containing three arrays explaining the classification decision",
                "properties": {
                    "primary_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Main reasons supporting the selected category"
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional evidence that strengthens the classification"
                    },
                    "contradicting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Evidence that contradicts the classification (use empty array if none)"
                    }
                },
                "required": ["primary_factors", "supporting_evidence", "contradicting_evidence"]
            }
        },
        "required": ["category", "confidence_score", "key_evidence", "reasoning"],
        "additionalProperties": False
    },
    
    formatting_rules=[
        "Output MUST be valid, parseable JSON",
        "Do not include any text before the opening { or after the closing }",
        "CRITICAL: Do NOT wrap JSON in markdown code blocks - no ``` or ```json markers",
        "CRITICAL: Output pure JSON starting with { and ending with } - nothing else",
        "Ensure all arrays and objects are properly closed",
        "Use empty arrays [] for missing values, not null or empty strings",
        "Do not include trailing commas",
        "Ensure proper escaping of special characters in strings",
        "",
        "Quote Handling - JSON Structure: ALWAYS use ASCII double quotes for JSON keys and string boundaries",
        "Quote Handling - Cited Content: When quoting text containing fancy Unicode quotes, replace them with ASCII apostrophes",
        "Fancy Unicode quotes to replace: German opening (U+201E), left/right double (U+201C/U+201D), all single quotes",
        "All fancy quotes become regular apostrophe (') when cited inside JSON string values",
        "",
        "Summary: Double quotes for JSON structure, apostrophes for fancy-quoted content inside strings"
    ],
    
    validation_requirements=[
        "Must be valid JSON format",
        "Category must match exactly from predefined list",
        "Confidence score must be number between 0.0 and 1.0",
        "All required fields must be present",
        "key_evidence and reasoning must be objects with nested arrays"
    ],
    
    evidence_validation_rules=[
        "Message Evidence must include direct quotes with speaker identification",
        "Shipping Evidence must include tracking events with timestamps",
        "Timeline Evidence must show chronological sequence of events",
        "All evidence must reference specific content from input data"
    ],
    
    example_output={
        "category": "TrueDNR",
        "confidence_score": 0.92,
        "key_evidence": {
            "message_evidence": [
                "[BUYER]: Hello, I have not received my package, but I see the order shows that it has been delivered, why?",
                "[BUYER]: But I did not find any package, please refund me, thank you"
            ],
            "shipping_evidence": [
                "[Event Time]: 2025-02-21T17:40:49.323Z [Ship Track Event]: Delivered to customer",
                "No further shipping events after delivery confirmation"
            ],
            "timeline_evidence": [
                "Delivery confirmation on 2025-02-21 17:40",
                "Buyer reports non-receipt starting 2025-02-25 07:14"
            ]
        },
        "reasoning": {
            "primary_factors": [
                "Tracking shows package was delivered successfully",
                "Buyer explicitly states they did not receive the package after delivery scan"
            ],
            "supporting_evidence": [
                "Buyer requests refund due to missing package",
                "No evidence of buyer receiving wrong/defective item"
            ],
            "contradicting_evidence": []
        }
    }
)

# Instruction Configuration (with detailed classification guidelines)
instruction_settings = InstructionConfig(
    include_analysis_steps=True,
    include_decision_criteria=True,
    include_reasoning_requirements=True,
    step_by_step_format=True,
    include_evidence_validation=True,
    
    classification_guidelines={
        "sections": [
            {
                "title": "## Classification Guidelines",
                "subsections": [
                    {
                        "title": "### 1. Output Format Requirements",
                        "content": [
                            "**Category Selection:**",
                            "- Choose exactly ONE category from the provided list",
                            "- Category name must match exactly (case-sensitive)",
                            "",
                            "**Confidence Score:**",
                            "- Provide as decimal number between 0.00 and 1.00 (e.g., 0.95)",
                            "- Base confidence for complete data: 0.7-1.0",
                            "- Missing one field: reduce by 0.1-0.2",
                            "- Missing two fields: reduce by 0.2-0.3",
                            "- Minimum confidence threshold: 0.5"
                        ]
                    },
                    {
                        "title": "### 2. Shiptrack Parsing Rules",
                        "content": [
                            "**Multiple Shipment Structure:**",
                            "- Multiple shipment sequences separated by shipment IDs",
                            "- Each sequence starts with \"[bom] [Shipment ID]:* [eom]\"",
                            "",
                            "**Analysis Approach:**",
                            "- Process each shipment sequence separately",
                            "- Compare delivery events (EVENT_301) across all sequences",
                            "",
                            "**Key Event Codes:**",
                            "- EVENT_301: Delivery confirmation",
                            "- EVENT_302: Out for delivery",
                            "- EVENT_201: Arrival at facility"
                        ]
                    },
                    {
                        "title": "### 3. Missing Data Handling",
                        "content": [
                            "**When Dialogue is Empty but Shiptrack Exists:**",
                            "- Focus on shipping events and timeline",
                            "- Reduce confidence score by 0.1-0.2",
                            "",
                            "**When Shiptrack is Empty but Dialogue Exists:**",
                            "- Focus on message content and reported issues",
                            "- Reduce confidence score by 0.1-0.2"
                        ]
                    },
                    {
                        "title": "### 4. Category Priority Hierarchy",
                        "content": [
                            "**Tier 1: Abuse Pattern Categories (Highest Priority)**",
                            "- PDA_Undeliverable: Verify no delivery + refund given",
                            "- PDA_Early_Refund: Verify refund before delivery",
                            "",
                            "**Tier 2: Delivery Status Categories**",
                            "- TrueDNR: Delivered but disputed",
                            "- Confirmed_Delay: External factors confirmed"
                        ]
                    },
                    {
                        "title": "### 5. Evidence Requirements",
                        "content": [
                            "**Message Evidence Must Include:**",
                            "- Direct quotes from dialogue with speaker identification",
                            "",
                            "**Shipping Evidence Must Include:**",
                            "- All tracking events listed chronologically",
                            "",
                            "**Timeline Evidence Must Show:**",
                            "- Clear chronological sequence of events"
                        ]
                    }
                ]
            }
        ]
    }
)

# Configure the step using DAG config factory
step_name = "BedrockPromptTemplateGeneration"
factory.set_step_config(
    step_name,
    
    # Input placeholders
    input_placeholders=['dialogue', 'shiptrack_event_history_by_order', 'shiptrack_max_estimated_arrival_date_by_order'],
    prompt_configs_path=str(category_definitions_path.parent),  # Directory containing all JSON configs
    
    # Basic settings
    template_task_type="buyer_seller_classification",
    template_style="structured",
    validation_level="comprehensive",
    template_version="2.0",
    
    # Output configuration
    output_format_type="structured_text",
    required_output_fields=["Category", "Confidence Score", "Key Evidence", "Reasoning"],
    
    # Template features
    include_examples=True,
    generate_validation_schema=True,
    
    # Sub-configurations (Pydantic models)
    system_prompt_settings=system_prompt_settings,
    output_format_settings=output_format_settings,
    instruction_settings=instruction_settings,
    
    processing_entry_point='bedrock_prompt_template_generation.py'
)
print(f"✅ {step_name} configured")
```

## Generated Prompt Structure

When the above configuration runs, it will generate a prompt template with this structure:

### System Prompt Section
```
You are an expert in analyzing buyer-seller messaging conversations and shipping logistics 
with extensive knowledge in buyer-seller messaging analysis, shipping logistics, delivery 
timing analysis, e-commerce dispute resolution, classification and categorization. Your task 
is to classify interactions based on message content, analyze shipping events and delivery 
timing, categorize into predefined dispute categories, provide evidence-based reasoning for 
classifications. Always be precise in classification decisions, be objective in evidence 
evaluation, be thorough in timeline analysis, follow exact formatting requirements, consider 
all available evidence sources in your analysis.
```

### User Prompt Template Structure

1. **Category Definitions** (13 categories with full details)
2. **Input Placeholders**
   - Dialogue: {dialogue}
   - Shiptrack: {shiptrack_event_history_by_order}
   - Estimated Delivery: {shiptrack_max_estimated_arrival_date_by_order}
3. **Analysis Instructions** (step-by-step)
4. **Classification Guidelines** (200+ lines covering):
   - Output format requirements
   - Shiptrack parsing rules
   - Missing data handling
   - Category priority hierarchy
   - Evidence requirements
5. **Required Output Format** (with examples and rules)

## Key Features Demonstrated

### 1. Modular Configuration
- Each component (system_prompt, output_format, instruction, categories) in separate JSON files
- Easy to update individual components without touching code

### 2. Structured JSON Output
- Valid, parseable JSON format for LLM responses
- Complete example showing exact expected structure
- Nested objects and arrays for complex data
- Schema-driven validation support

### 3. Detailed Guidelines
- 200+ lines of classification guidance
- Hierarchical structure (sections → subsections → content)
- Fully data-driven from JSON config
- JSON-specific formatting instructions

### 4. Evidence Validation
- Three evidence types (Message, Shipping, Timeline)
- Specific formatting requirements
- Validation rules for each type
- Example output showing correct JSON structure

## Benefits of Enhanced Approach

✅ **Zero Hard-Coding**: All prompt content comes from JSON configs  
✅ **Flexible Format**: Support both structured_json and structured_text  
✅ **Rich Customization**: Add detailed guidelines without code changes  
✅ **Reusable**: Share configs across similar classification tasks  
✅ **Maintainable**: Update categories or guidelines by editing JSON  
✅ **Testable**: Validate configs with Pydantic models  

## Next Steps

1. Create the 4 JSON config files in your prompt_configs directory
2. Load category definitions from `category_definitions.json`
3. Configure the step in your DAG config as shown above
4. Run the prompt template generation step
5. The system will auto-generate:
   - `prompts.json` (main template)
   - `template_metadata_*.json` (generation metadata)
   - `validation_schema_*.json` (response validation schema)

The generated prompt will match your target format exactly, with all 13 categories, structured text output, and comprehensive classification guidelines!
