# Technical Design Documentation
## Synthetic Radio Host - Hinglish Podcast Generator

---

## Table of Contents
1. [One-Shot Inference Model](#one-shot-inference-model)
2. [Accent Tag System](#accent-tag-system)
3. [System Architecture](#system-architecture)
4. [Prompt Engineering Strategy](#prompt-engineering-strategy)

---

## One-Shot Inference Model

### Overview
The system employs a **one-shot learning approach** (also known as one-shot inference) to guide the LLM in generating authentic Hinglish conversations. This technique provides a single, concrete example within the prompt to demonstrate the desired output style, enabling the model to understand and replicate the conversational pattern without requiring fine-tuning or multiple examples.

### Implementation Details

#### Location in Code
**File:** `syntheticRadioHostScript.py`  
**Function:** `create_hinglish_prompt()`  
**Lines:** 184-190

#### How It Works

1. **Example Inclusion**: The prompt includes a complete example conversation under the section `### EXAMPLE (ONE-SHOT) ###`:

```python
### EXAMPLE (ONE-SHOT) ###
Topic: Rain in Mumbai
Vineet: [pause] Swagat hai doston! aaj Mumbai ki baarish ke baare main baat karate hain. 
        Rains have finallyarrived, hai na? [laughs]
Simran: Oh my god, Literally... it's so beautiful but my hair is like... 
        pura mess ho gaya hai, you know?
Vineet: Arre, baalon ko chhodo na! Just look at the cutting chai and vada pav weather... 
        bindaas enjoy karo!
Simran: I mean... [pause] true. It is a total vibe. Par ye traffic... hmmm... 
        it's going to be insane, right?
Vineet: Wo toh hai... par Mumbai ki baarish is emotion, Simran. Bilkul tension mat lo... 
        bill-kul chill maaro and ejoy the rains!
```

2. **Pattern Demonstration**: The example demonstrates:
   - **Phonetic spelling**: "bill-kul" instead of "bilkul"
   - **Code-switching**: Mixing Hindi and English naturally
   - **Emotional expressions**: `[laughs]`, `[pause]`
   - **Conversational fillers**: "umm...", "like...", "hmmm"
   - **Indian-isms**: "...hai na?", "...yaarrr"
   - **Natural interruptions**: Using "..." to show dialogue flow

3. **Task Specification**: After the example, the prompt instructs:
   ```
   ### YOUR TASK ###
   Now, generate a similar natural sounding 2-minute conversation about the topic: {topic}
   Using this context: {context}
   ```

### Why One-Shot Learning?

#### Advantages:
1. **No Fine-Tuning Required**: Works with pre-trained models (Ollama's llama3.2) without additional training
2. **Flexible**: Can adapt to any topic by changing the context while maintaining style consistency
3. **Cost-Effective**: Single inference call, no training data needed
4. **Rapid Prototyping**: Easy to modify the example to change conversation style
5. **Consistency**: The example ensures the model understands the exact format and style expected

#### Technical Benefits:
- **Few-Shot Learning**: Demonstrates the pattern in-context
- **Style Transfer**: Transfers the conversational style from the example to new topics
- **Format Compliance**: Ensures the output follows the exact dialogue format shown

### Prompt Structure

The one-shot prompt follows this structure:

```
1. Task Description
   └─ "Write a high-energy, 2-minute Radio Duo conversation..."

2. Constraints & Instructions
   └─ Duration, word count, formatting rules

3. Style Guidelines
   └─ Phonetic spelling, fillers, emotional expressions

4. ONE-SHOT EXAMPLE ⭐
   └─ Complete example conversation demonstrating all patterns

5. Task Assignment
   └─ "Now generate a similar conversation about {topic}"
```

### Code Reference

```python
def create_hinglish_prompt(topic, context):
    prompt = f"""
    Write a high-energy, 2-minute "Radio Duo" conversation in Hinglish...
    
    [Instructions and constraints...]
    
    ### EXAMPLE (ONE-SHOT) ###
    Topic: Rain in Mumbai
    Vineet: [pause] Swagat hai doston!...
    [Complete example dialogue...]
    
    ### YOUR TASK ###
    Now, generate a similar natural sounding 2-minute conversation about the topic: {topic}
    Using this context: {context}
    """
    return prompt
```

---

## Accent Tag System

### Overview
The system implements an **alternating accent tag mechanism** to enhance the naturalness and authenticity of the generated audio. Each dialogue segment is prefixed with either `[Indian English]` or `[Hindi accent]` tags, which guide the ElevenLabs TTS engine to apply appropriate pronunciation and intonation patterns.

### Implementation Details

#### Location in Code
**File:** `syntheticRadioHostScript.py`  
**Function:** `parse_script()`  
**Lines:** 397-404

#### How It Works

1. **Tag Assignment Logic**: Tags are assigned in an alternating pattern based on dialogue index:

```python
# Add alternating tags at the start of each dialogue
# Even indices (0, 2, 4, ...) get [Indian English]
# Odd indices (1, 3, 5, ...) get [Hindi accent]
if dialogue_index % 2 == 0:
    text = f"[Indian English] {text}"
else:
    text = f"[Hindi accent] {text}"
dialogue_index += 1
```

2. **Tag Distribution**:
   - **Even-indexed dialogues** (0, 2, 4, 6...): `[Indian English]`
   - **Odd-indexed dialogues** (1, 3, 5, 7...): `[Hindi accent]`

3. **Integration with TTS**: The tags are sent directly to ElevenLabs API:

```python
def generate_segment(text, api_key, voice_id, use_emotions=True):
    # text already contains [Indian English] or [Hindi accent] tag
    data = {
        "text": text,  # e.g., "[Indian English] Swagat hai doston!"
        "model_id": "eleven_v3",
        "voice_settings": {...}
    }
    response = requests.post(url, json=data, headers=headers)
```

### Tag Definitions

#### `[Indian English]`
- **Purpose**: Indicates Indian English pronunciation patterns
- **Characteristics**:
  - English words with Indian accent
  - Natural code-switching between languages
  - Indian English intonation patterns
  - Slight Indian accent on English words

#### `[Hindi accent]`
- **Purpose**: Indicates stronger Hindi accent/influence
- **Characteristics**:
  - More pronounced Hindi pronunciation
  - Stronger Indian accent
  - Natural Hindi intonation
  - Authentic Hindi word pronunciation

### Why Alternating Tags?

#### Design Rationale:
1. **Natural Variation**: Alternating accents create a more natural, dynamic conversation
2. **Voice Distinction**: Helps differentiate between hosts even when using similar voice models
3. **Authentic Hinglish**: Reflects real-world Hinglish conversations where speakers naturally vary their accent
4. **TTS Optimization**: Guides the TTS engine to apply appropriate pronunciation rules

### Example Output

**Before Tag Addition:**
```
Vineet: Swagat hai doston! Aaj Mumbai Indians ke baare main baat karte hain.
Simran: Oh my god, literally... they're so amazing!
```

**After Tag Addition:**
```
[Indian English] Swagat hai doston! Aaj Mumbai Indians ke baare main baat karte hain.
[Hindi accent] Oh my god, literally... they're so amazing!
```

### Integration Flow

```
Generated Script
    ↓
parse_script()
    ↓
Dialogue Segments
    ↓
Alternating Tag Assignment
    ├─ Segment 0: [Indian English] ...
    ├─ Segment 1: [Hindi accent] ...
    ├─ Segment 2: [Indian English] ...
    └─ Segment 3: [Hindi accent] ...
    ↓
generate_segment() → ElevenLabs API
    ↓
Audio with Natural Accent Variation
```

### Code Reference

```python
def parse_script(script_text, male_host_name="Host1", female_host_name="Host2"):
    dialogue = []
    dialogue_index = 0  # Track index for alternating tags
    
    for segment in segments:
        # ... parsing logic ...
        
        # Add alternating tags at the start of each dialogue
        if dialogue_index % 2 == 0:
            text = f"[Indian English] {text}"
        else:
            text = f"[Hindi accent] {text}"
        dialogue_index += 1
        
        dialogue.append((speaker, speaker_name, text))
    
    return dialogue
```

### Benefits

1. **Enhanced Naturalness**: Creates more authentic-sounding conversations
2. **Better Pronunciation**: TTS engine applies appropriate accent rules
3. **Voice Variation**: Adds subtle variation even with same voice model
4. **Cultural Authenticity**: Reflects real Hinglish conversation patterns
5. **Flexible Control**: Easy to modify tag assignment logic if needed

---

## System Architecture

### High-Level Flow

```
┌─────────────────┐
│  Wikipedia API  │
└────────┬────────┘
         │ Context
         ▼
┌─────────────────┐
│  Prompt Builder │ ◄── One-Shot Example
│  (with Example) │
└────────┬────────┘
         │ Prompt
         ▼
┌─────────────────┐
│  Ollama LLM     │
│  (llama3.2)     │
└────────┬────────┘
         │ Script
         ▼
┌─────────────────┐
│  Script Parser  │ ◄── Accent Tag Assignment
│  (with Tags)    │
└────────┬────────┘
         │ Tagged Dialogue
         ▼
┌─────────────────┐
│ ElevenLabs TTS  │
└────────┬────────┘
         │ Audio Segments
         ▼
┌─────────────────┐
│ Audio Combiner  │
└────────┬────────┘
         │
         ▼
    Final MP3
```

### Component Interaction

1. **Context Fetching** → Wikipedia provides topic context
2. **Prompt Generation** → One-shot example embedded in prompt
3. **LLM Inference** → Single-shot generation based on example
4. **Script Parsing** → Accent tags added to each segment
5. **Audio Synthesis** → TTS applies accent tags for pronunciation
6. **Audio Assembly** → Segments combined into final podcast

---

## Prompt Engineering Strategy

### Multi-Layered Approach

The prompt uses a **three-layer strategy** to ensure high-quality output:

1. **Explicit Instructions**: Detailed rules and constraints
2. **One-Shot Example**: Concrete demonstration of desired output
3. **Context Injection**: Wikipedia context for factual accuracy

### Key Design Principles

1. **Specificity**: Clear, unambiguous instructions
2. **Exemplification**: One concrete example showing all patterns
3. **Constraints**: Strict formatting and duration requirements
4. **Cultural Context**: Indian-isms and Hinglish patterns
5. **Naturalness**: Emphasis on conversational elements (fillers, emotions)

### Prompt Components

```
┌─────────────────────────────────────┐
│ 1. Task Definition                  │
│    "Write a 2-minute conversation"   │
├─────────────────────────────────────┤
│ 2. Constraints                      │
│    - 14 lines total                 │
│    - 300-400 words                  │
│    - Specific format                │
├─────────────────────────────────────┤
│ 3. Style Guidelines                 │
│    - Phonetic spelling              │
│    - Fillers & emotions             │
│    - Code-switching patterns        │
├─────────────────────────────────────┤
│ 4. ONE-SHOT EXAMPLE ⭐              │
│    Complete example conversation    │
│    Demonstrating all patterns       │
├─────────────────────────────────────┤
│ 5. Context & Task                   │
│    Topic: {topic}                   │
│    Context: {context}               │
└─────────────────────────────────────┘
```

---

## Technical Specifications

### One-Shot Inference
- **Model**: Ollama llama3.2
- **Inference Type**: Single forward pass (no fine-tuning)
- **Example Count**: 1 (one-shot)
- **Example Length**: ~5 dialogue exchanges
- **Token Efficiency**: Minimal prompt overhead

### Accent Tag System
- **Tag Types**: 2 (`[Indian English]`, `[Hindi accent]`)
- **Assignment**: Alternating pattern
- **TTS Model**: ElevenLabs v3 (supports accent tags)
- **Tag Position**: Prefix to dialogue text
- **Processing**: Applied during script parsing phase

---

## Future Enhancements

### One-Shot Model
- [ ] Support for multiple example styles
- [ ] Dynamic example selection based on topic
- [ ] Few-shot learning option (2-3 examples)

### Accent Tags
- [ ] Custom accent tag definitions
- [ ] Per-host accent assignment (not just alternating)
- [ ] Accent intensity control
- [ ] Regional accent variations (Mumbai, Delhi, etc.)

---

## References

- **One-Shot Learning**: Brown et al., "Language Models are Few-Shot Learners" (GPT-3 paper)
- **Prompt Engineering**: Best practices for in-context learning
- **ElevenLabs TTS**: Documentation on accent tag support
- **Hinglish Linguistics**: Code-switching patterns in Indian English

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: Synthetic Radio Host Development Team

