"""
Synthetic Radio Host - Generate Hinglish Podcast Scripts
Converts Wikipedia topics into natural Hinglish radio conversations using Ollama LLM
and generates audio using ElevenLabs API.
"""

import wikipediaapi
import ollama
import os
import requests
import time
import re
import sys
from dotenv import load_dotenv


# Get ElevenLabs API key from command-line argument or environment variable
def get_elevenlabs_api_key():
    """
    Get ElevenLabs API key from command-line argument or environment variable.
    
    Priority:
    1. Command-line argument (if provided)
    2. Environment variable from .env file
    
    Returns:
        API key string or None if not found
    """
    # Check if API key is provided as command-line argument (sys.argv[1])
    if len(sys.argv) > 1:
        api_key = sys.argv[1].strip()
        if api_key:
            print("✅ Using ElevenLabs API key from command-line argument")
            return api_key
    
    print(" \n\n ElevenLabs API key not present in command-line argument\n")

    print("Usage is like this: python syntheticRadioHostScript.py <api_key> <output_file_path>")
    print("Will read api key from environment variable .env file whose location is C:/Users/vineet.srivastava/venvGenAIStudy/.env .\n\n")    
    
    # Ask user if they want to proceed with environment variable
    user_input = input("Do you want to proceed with reading API key from .env file? (Y/N): ").strip()
    if user_input.lower() != 'y':
        print("❌ Exiting. Please provide API key as command-line argument or set it in .env file in the path mentioned.")
        sys.exit(0)
    
    # Fall back to environment variable
    # Load environment variables
    load_dotenv("C:/Users/vineet.srivastava/venvGenAIStudy/.env")
    api_key = os.getenv("ELLEVENLABS_API_KEY")
    if api_key:
        print("✅ Using ElevenLabs API key from environment variable")
    else:
        print("⚠️ No ElevenLabs API key found (neither in command-line nor environment)")
    
    return api_key


def get_output_file_path():
    """
    Get output file path from command-line argument.
    
    Returns:
        Output file path string or None if not provided
    """
    # Check if output file path is provided as command-line argument (sys.argv[2])
    if len(sys.argv) > 2:
        output_path = sys.argv[2].strip()
        if output_path:
            print(f"✅ Using output file path from command-line argument: {output_path}")
            return output_path
    
    # If not provided, return None (will use default in main())
    return None

ELEVEN_LABS_API_KEY = get_elevenlabs_api_key()


def fetch_wikipedia_context(topic, max_chars=2000):
    """
    Fetch context from Wikipedia for a given topic
    
    Args:
        topic: Topic to search for
        max_chars: Maximum characters to return
    
    Returns:
        Context string or None if not found
    """
    if not topic or not topic.strip():
        print("❌ Topic is empty or invalid.")
        return None
    
    try:
        wiki = wikipediaapi.Wikipedia('SyntheticRadio/1.0', 'en')
        page = wiki.page(topic)
        
        if not page.exists():
            print(f"❌ Topic '{topic}' not found on Wikipedia.")
            return None
        
        context = page.summary[:max_chars] if page.summary else None
        
        if not context or len(context.strip()) < 50:
            print(f"⚠️ Warning: Context for '{topic}' is too short or empty ({len(context) if context else 0} chars).")
            print("   Script generation may be limited.")
            return context if context else None
        
        return context
    except Exception as e:
        print(f"❌ Error fetching Wikipedia context: {e}")
        return None


def clean_script_markdown(script_text):
    """
    Clean markdown formatting from LLM-generated script
    
    Args:
        script_text: Raw script text with markdown
    
    Returns:
        Cleaned script text
    """
    # Remove markdown bold (**text**)
    script_text = re.sub(r'\*\*(.*?)\*\*', r'\1', script_text)
    
    # Remove markdown headers (###, ##, #)
    script_text = re.sub(r'^#+\s*', '', script_text, flags=re.MULTILINE)
    
    # Remove theme music markers
    script_text = re.sub(r'\*\*\[Theme music.*?\]\*\*', '', script_text, flags=re.IGNORECASE)
    script_text = re.sub(r'\[Theme music.*?\]', '', script_text, flags=re.IGNORECASE)
    
    # Remove note sections at the end
    script_text = re.sub(r'Note:.*$', '', script_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    script_text = re.sub(r'\n{3,}', '\n\n', script_text)
    
    return script_text.strip()


def create_hinglish_prompt(topic, context):
    """
    Create the prompt for generating Hinglish radio conversation
    
    Args:
        topic: Topic to discuss
        context: Wikipedia context about the topic
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""
    Write a high-energy, 2-minute "Radio Duo" conversation in Hinglish between two hosts, 'Vineet' (energetic) and 'Simran' (curious) about: {topic}.

    CRITICAL INSTRUCTIONS FOR NATURAL SOUND:
    1. PHONETIC SPELLING: Write Hindi words exactly how they sound to an English speaker. 
       (e.g., instead of 'achcha', write 'ach-chaaa' for emphasis.).
    2. Scatter 'umm...', 'uhh', 'hmmm', and 'like...' frequently. 
    3. Use "..." at the end of a line and the start of the next to show Host B cutting in.
    4. Use bracketed cues like "[laughs]", "[sighs]", "[giggles]" in mid of sentences to keep conversation lively as per the sentence in between in each podcast strictly.
    5. THE "HINGLISH" FLOW: Use 'Indian-isms' like:
       - Ending English sentences with: "...na?", "...yaarrr", or "...hai na?"
       - Starting sentences with: "Look...", "Suno...", "Listen..."
    6. TEXT STRUCTURE: Keep sentences short. Long sentences sound robotic. Short, punchy sentences sound like a real chat.
    7. Start with Vineet starting the show greeting the audience.
    8. Keep pauses in middle of each sentences using [pause] only.
    9. Vineet has to sound like a Host with super energetic tone with touch of Mumbai language.
    10. Conversation must be engaging, fun, and full of energy.
    11. Should not have any Radio station name in conversation.
    12. End each statement of host with split marker "\n\n" only.
    13. 2017 should sound as two thousand seventeen and not two thousand and seventeen.
    14. Conversation should be 2 minutes long strictly.
    
    STRICT FORMAT:
    Vineet: [text]
    Simran: [text]

    ### EXAMPLE (ONE-SHOT) ###
    Topic: Rain in Mumbai
    Vineet: [pause] Swagat hai doston! Vineet here... aaj Mumbai ki baarish ke baare main baat karate hain. Rains have finallyarrived, hai na? [laughs]
    Simran: Oh my god, Literally... it's so beautiful but my hair is like... pura mess ho gaya hai, you know?
    Vineet: Arre Simran, baalon ko chhodo na! Just look at the cutting chai and vada pav weather... bindaas enjoy karo!
    Simran: I mean... [pause] true. It is a total vibe. Par ye traffic... hmmm... it's going to be insane, right?
    Vineet: Wo toh hai... par Mumbai ki baarish is emotion, Simran. Bilkul tension mat lo... bill-kul chill maaro!
    
    ### YOUR TASK ###
    Now, generate a similar 2-minute conversation about the topic: {topic}
    Using this context: {context}
    Output irectly from script without any other text.
    """
    return prompt


def check_ollama_connection():
    """
    Check if Ollama is running by attempting to list models
    
    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        # Try to list models to check if Ollama is running
        models_response = ollama.list()
        if models_response is None:
            print("❌ Cannot connect to Ollama: Response is None")
            print("   Make sure Ollama is running: ollama serve")
            return False
        
        # If we get any response (not None), Ollama is running
        print("✅ Ollama is running and accessible")
        return True
        
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("   Make sure Ollama is running: ollama serve")
        return False


def generate_script_with_ollama(topic, context, model='llama3.2'):
    """
    Generate Hinglish radio script using Ollama LLM
    
    Args:
        topic: Topic to discuss
        context: Wikipedia context
        model: Ollama model to use
    
    Returns:
        Tuple of (Generated script text or None if failed, time_taken in seconds)
    """
    if not context or len(context.strip()) < 20:
        print("⚠️ Warning: Context is very short. Script quality may be affected.")
    
    print(f"\ngenerate_script_with_ollama: topic is {topic} and context is {context}")
    prompt = create_hinglish_prompt(topic, context)
    
    print(f"\n\n🤖 Generating 2 minute script with Ollama ({model})...")
    
    try:
        # Record start time
        start_time = time.time()
        
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': 'You are a creative radio scriptwriter specialized in Hinglish.'},
            {'role': 'user', 'content': prompt},
        ])
        
        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        script = response['message']['content']
        
        if not script or len(script.strip()) < 50:
            print("⚠️ Warning: Generated script is very short or empty.")
            return None, elapsed_time
        
        # Clean markdown from the script
        #script = clean_script_markdown(script)
        
        #if not script or len(script.strip()) < 50:
        #    print("⚠️ Warning: Script after cleaning is very short or empty.")
        #    return None, elapsed_time
        
        print("✅ Script generated successfully!")
        print(f"script has {len(script.split())} words")
        print("Script is: ", script)
        return script, elapsed_time
        
    except Exception as e:
        print(f"❌ Error generating script with Ollama: {e}")
        return None, 0


def parse_script(script_text, male_host_name="Host1", female_host_name="Host2"):
    """
    Parse script text into structured dialogue
    
    Script format (separated by \\n\\n or single \\n):
    Vineet: Hello everyone! [laughs]\\n\\nSimran: Welcome back!\\n\\nVineet: Next line
    
    Args:
        script_text: Raw script with separators
        male_host_name: Name of male host in script (default: "Host1")
        female_host_name: Name of female host in script (default: "Host2")
    
    Returns list of tuples: [(speaker, name, text), ...]
    """
    
    if not script_text:
        return []
    
    # First, try to extract just the dialogue part (remove any intro/outro text)
    # Look for patterns like "Vineet:" or "Simran:" to find where dialogue starts
    # Handle markdown in speaker names (e.g., **John**: or *Mary*:)
    lines = script_text.split('\n')
    dialogue_start_idx = 0
    for i, line in enumerate(lines):
        # Remove markdown from line for pattern matching
        line_clean = re.sub(r'[\[\]()*]', '', line)
        # Check if line contains a speaker name followed by colon (with or without markdown)
        if re.search(rf'\b({male_host_name}|{female_host_name}):', line_clean, re.IGNORECASE):
            dialogue_start_idx = i
            break
    
    # Extract dialogue portion
    dialogue_text = '\n'.join(lines[dialogue_start_idx:])
    
    # Try multiple splitting strategies
    # Strategy 1: Split by double newline
    segments_double = [s.strip() for s in dialogue_text.split('\n\n') if s.strip()]
    
    # Strategy 2: Split by single newline when line starts with speaker name
    segments_single = []
    current_segment = ""
    for line in dialogue_text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            # Empty line - end current segment if exists
            if current_segment:
                segments_single.append(current_segment)
                current_segment = ""
            continue
        
        # Check if line starts with a speaker name pattern (with colon)
        # Pattern: "SpeakerName:" or "SpeakerName: text" (handle markdown like **John**: or *Mary*:)
        # First remove markdown for pattern matching
        line_clean_for_match = re.sub(r'[\[\]()*]', '', line_stripped)
        speaker_pattern = rf'^\s*({male_host_name}|{female_host_name}):\s*'
        if re.match(speaker_pattern, line_clean_for_match, re.IGNORECASE):
            # New speaker line found
            if current_segment:
                segments_single.append(current_segment)
            current_segment = line_stripped
        else:
            # Continuation of current segment (multi-line dialogue)
            if current_segment:
                current_segment += " " + line_stripped
            else:
                # Line doesn't start with speaker but might be part of dialogue
                # Check if it has a colon pattern (might be a different speaker name)
                if ':' in line_stripped and re.search(r'^[^:]+:\s*.+', line_stripped):
                    # Looks like a speaker line with different name
                    if current_segment:
                        segments_single.append(current_segment)
                    current_segment = line_stripped
                else:
                    # Just add to current segment
                    current_segment = line_stripped
    
    # Add last segment
    if current_segment:
        segments_single.append(current_segment)
    
    # Choose the best splitting strategy
    # Prefer double newline if it yields reasonable segments (>= 3)
    # Otherwise use single newline splitting
    if len(segments_double) >= 3:
        segments = segments_double
        print(f"parse_script: Using double newline splitting ({len(segments)} segments)")
    else:
        segments = segments_single
        print(f"parse_script: Using single newline splitting ({len(segments)} segments)")
    
    dialogue = []
    dialogue_index = 0  # Track index for alternating tags
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Check if segment has speaker label (Name: text)
        # Look for pattern: speaker_name: text
        # Handle both single-line and multi-line segments
        colon_match = re.search(r'^([^:]+?):\s*(.+)$', segment, re.DOTALL)
        
        if colon_match:
            speaker_name = colon_match.group(1).strip()
            text = colon_match.group(2).strip()
            
            # Remove any remaining markdown or formatting from speaker name
            speaker_name = re.sub(r'[\[\]()*]', '', speaker_name).strip()
            text = text.strip()
            
            if not text:  # Skip if no text after colon
                continue
            
            # Add alternating tags at the start of each dialogue
            # Even indices (0, 2, 4, ...) get [Indian English]
            # Odd indices (1, 3, 5, ...) get [Hindi accent]
            if dialogue_index % 2 == 0:
                text = f"[Indian English] {text}"
            else:
                text = f"[Hindi accent] {text}"
            dialogue_index += 1
            
            # Determine if male or female host
            speaker_lower = speaker_name.lower()
            male_lower = male_host_name.lower()
            female_lower = female_host_name.lower()
            
            if speaker_lower == male_lower or speaker_name == male_host_name:
                dialogue.append(('host1', speaker_name, text))
            elif speaker_lower == female_lower or speaker_name == female_host_name:
                dialogue.append(('host2', speaker_name, text))
            else:
                # Try to detect by common names or patterns
                # If not found, treat first unique name as host1, second as host2
                if not dialogue:
                    dialogue.append(('host1', speaker_name, text))
                else:
                    # Check if this speaker already appeared
                    existing_speakers = {d[1]: d[0] for d in dialogue}
                    if speaker_name in existing_speakers:
                        dialogue.append((existing_speakers[speaker_name], speaker_name, text))
                    else:
                        # New speaker - assign opposite of last speaker
                        last_speaker = dialogue[-1][0]
                        new_speaker = 'host2' if last_speaker == 'host1' else 'host1'
                        dialogue.append((new_speaker, speaker_name, text))
        else:
            # No label - try to add to last speaker if it looks like continuation
            if dialogue and len(segment) > 10:  # Only if substantial text
                last_speaker, last_name, last_text = dialogue[-1]
                # Note: Continuation text doesn't get a new tag, it's part of the previous dialogue
                dialogue[-1] = (last_speaker, last_name, last_text + " " + segment)
    
    print(f"parse_script: Parsed {len(dialogue)} dialogue segments")
    return dialogue


def generate_segment(text, api_key, voice_id, use_emotions=True):
    """
    Generate a single audio segment for one speaker
    
    Args:
        text: The text to convert (supports emotion tags like [laughs])
        api_key: ElevenLabs API key
        voice_id: Voice ID to use
        use_emotions: Whether to use v3 model for emotion support
    
    Returns:
        Audio bytes or None if failed
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    # Use v3 for emotion tags, turbo for speed
    model = "eleven_v3"
    
    # Check if text contains emotion tags
    emotion_tags = ['[laughs]', '[giggles]', '[sighs]', '[excited]', '[whispers]', '[shouts]']
    has_emotions = any(tag in text for tag in emotion_tags)
    
    data = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": 0.5,  # Lower for more natural variation
            "similarity_boost": 0.75,
            "style": 0.5 if has_emotions else 0.3,
            "use_speaker_boost": True
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            print(f"❌ Error generating segment: {response.status_code}")
            print(f"Message: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def validate_output_path(output_file):
    """
    Validate and create output directory if needed
    
    Args:
        output_file: Output file path
    
    Returns:
        True if valid, False otherwise
    """
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"📁 Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Check write permissions
        test_file = output_file + ".test"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except PermissionError:
            print(f"❌ No write permission for: {output_file}")
            return False
        except Exception as e:
            print(f"❌ Cannot write to output location: {e}")
            return False
    except Exception as e:
        print(f"❌ Error validating output path: {e}")
        return False


def generate_podcast(dialogue, api_key, male_voice_id, female_voice_id, 
                     output_file="podcast_output.mp3"):
    """
    Generate complete podcast from parsed dialogue
    
    Args:
        dialogue: List of tuples [(speaker, name, text), ...] from parse_script
        api_key: ElevenLabs API key
        male_voice_id: Voice ID for male host
        female_voice_id: Voice ID for female host
        output_file: Output filename
    
    Returns:
        True if successful
    """
    
    if not api_key or api_key == "your_api_key_here":
        print("⚠️ Please set your ELEVEN_LABS_API_KEY first!")
        return False
    
    if not dialogue:
        print("❌ No dialogue provided. Cannot generate podcast.")
        return False
    
    # Validate output path
    if not validate_output_path(output_file):
        return False
    
    # Validate voice IDs
    if not male_voice_id or not female_voice_id:
        print("❌ Invalid voice IDs provided.")
        return False
    
    print("🎙️ Starting podcast generation...")
    print("=" * 60)
    
    print(f"📝 Found {len(dialogue)} dialogue segments")
    
    # Show detected hosts
    host1_name = next((d[1] for d in dialogue if d[0] == 'host1'), "Host1")
    host2_name = next((d[1] for d in dialogue if d[0] == 'host2'), "Host2")
    print(f"🎤 Male Host: {host1_name}")
    print(f"🎤 Female Host: {host2_name}")
    print()
    
    # Generate and save individual segments
    temp_files = []
    
    for idx, (speaker, speaker_name, text) in enumerate(dialogue, 1):
        voice_id = male_voice_id if speaker == 'host1' else female_voice_id
        
        print(f"🎵 [{idx}/{len(dialogue)}] {speaker_name}")
        print(f"   Text: {text[:70]}{'...' if len(text) > 70 else ''}")
        
        audio_bytes = generate_segment(text, api_key, voice_id)
        
        if audio_bytes:
            temp_filename = f"C:/LEARNING/LANGCHAIN/temp_segment_{idx:03d}.mp3"
            try:
                # Ensure temp directory exists
                temp_dir = os.path.dirname(temp_filename)
                if temp_dir and not os.path.exists(temp_dir):
                    os.makedirs(temp_dir, exist_ok=True)
                
                with open(temp_filename, 'wb') as f:
                    f.write(audio_bytes)
                temp_files.append(temp_filename)
                print(f"   ✅ Success (saved to {temp_filename})")
            except Exception as e:
                print(f"   ❌ Failed to save temp file: {e}")
        else:
            print(f"   ❌ Failed to generate")
        
        print()
        time.sleep(0.5)  # Rate limiting
    
    # Combine all files
    if temp_files:
        print("💾 Combining all segments into final podcast...")
        with open(output_file, 'wb') as outfile:
            for temp_file in temp_files:
                with open(temp_file, 'rb') as infile:
                    outfile.write(infile.read())
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        print("=" * 60)
        print(f"✅ Podcast generated successfully!")
        print(f"📁 File: {output_file}")
        print(f"📝 Combined {len(temp_files)} segments")
        print()
        return True
    else:
        print("❌ No segments were generated successfully!")
        return False


# Voice IDs for ElevenLabs
VOICES = {
    "Vikram": "hczKB0VbXLcBTn17ShYS",
    "Reyanshi": "UrB5rVw5j9MDZWDZJtOJ",
    "Mihir": "fZpEbMnaJhQJwDY7lFcw",
    "Nikita": "SZfY4K69FwXus87eayHK",
    "Aura":"QbPozfpTbDrP62pzQpyg"
}


def main():
    """Main execution function"""
    # Configuration
    topic = "Mumbai Indians"
    ollama_model = "llama3.2"
    
    # Get output file path from command-line argument or use default
    output_file = get_output_file_path()
    if not output_file:
        output_file = "C:/LEARNING/LANGCHAIN/ollama_podcast.mp3"
        print(f"\n⚠️ No output file path provided. Using default: {output_file}")
        print()
    
    # Voice selection - validate voice IDs exist
    if "Mihir" not in VOICES or "Aura" not in VOICES:
        print("❌ Required voice IDs not found in VOICES dictionary.")
        return
    
    MALE_VOICE_ID = VOICES["Mihir"]
    FEMALE_VOICE_ID = VOICES["Aura"]
    
    # Host names
    MALE_HOST_NAME = "Vineet"
    FEMALE_HOST_NAME = "Simran"
    
    print("=" * 60)
    print("🎙️  SYNTHETIC RADIO HOST - HINGLISH PODCAST GENERATOR")
    print("=" * 60)
    print(f"📌 Topic: {topic}")
    print(f"📁 Output File: {output_file}")
    print()
    
    # Step 0: Validate topic
    if not topic or not topic.strip():
        print("❌ Topic is empty. Please provide a valid topic.")
        return
    
    # Step 0.5: Check Ollama connection
    print("🔍 Checking Ollama connection...")
    if not check_ollama_connection():
        print("❌ Ollama connection check failed. Exiting.")
        return
    print()
    
    # Step 1: Fetch Wikipedia context
    print("📚 Fetching Wikipedia context...")
    context = fetch_wikipedia_context(topic)
    if not context:
        print("❌ Failed to fetch context. Exiting.")
        return
    
    print(f"✅ Context fetched ({len(context)} characters)")
    print()
    
    # Step 2: Generate script with Ollama
    script, time_taken = generate_script_with_ollama(topic, context, model=ollama_model)
    if not script:
        print("❌ Failed to generate script. Exiting.")
        return
    
    # Print time taken
    minutes = int(time_taken // 60)
    seconds = int(time_taken % 60)
    if minutes > 0:
        print(f"⏱️  Time taken to generate script: {minutes} minute(s) {seconds} second(s)")
    else:
        print(f"⏱️  Time taken to generate script: {seconds} second(s)")
    print()
    
    print("📝 Generated Script Preview:")
    print("-" * 60)
    print(script[:500] + "..." if len(script) > 500 else script)
    print("-" * 60)
    print()
    
    # Step 3: Parse the script
    print("📋 Parsing script into dialogue segments...\n")
    print(f"   Looking for hosts: {MALE_HOST_NAME} and {FEMALE_HOST_NAME}")
    dialogue = parse_script(script, 
                           male_host_name=MALE_HOST_NAME, 
                           female_host_name=FEMALE_HOST_NAME)
    
    if not dialogue:
        print("❌ No dialogue found in script! Cannot generate podcast.")
        print("   Debug: Script length:", len(script), "characters")
        print("   Debug: First 200 chars of script:")
        print("   " + script[:200])
        return
    
    if len(dialogue) < 3:
        print(f"⚠️ Warning: Only {len(dialogue)} dialogue segments found. Expected more.")
        print("   This might indicate a parsing issue. Check script format.")
        print("   Debug: First few segments found:")
        for i, (speaker, name, text) in enumerate(dialogue[:3], 1):
            print(f"   {i}. {name}: {text[:50]}...")
    
    print(f"✅ Parsed {len(dialogue)} dialogue segments")
    print("dialogue: ", dialogue)
    print()
  
      # Ask user for confirmation before generating audio files
    print("=" * 60)
    user_input = input("🎵 Do you want to generate audio files? (Y/N): ").strip()
    
    if user_input.lower() != 'y':
        print("❌ Audio generation cancelled by user. Exiting from Program. BYE!!!")
        return True # Exit from the program
    
    print("✅ Proceeding with audio generation...")
    print()
      
    # Step 4: Generate podcast audio
    if not ELEVEN_LABS_API_KEY:
        print("⚠️ ELEVEN_LABS_API_KEY not found.")
        print("   Please provide it as:")
        print("   1. Command-line argument: python syntheticRadioHostScript.py <api_key> <output_file_path>")
        print("   2. Or set it in your .env file as ELLEVEENLABS_API_KEY")
        print("   Skipping audio generation.")
        return
    
    success = generate_podcast(
        dialogue=dialogue,
        api_key=ELEVEN_LABS_API_KEY,
        male_voice_id=MALE_VOICE_ID,
        female_voice_id=FEMALE_VOICE_ID,
        output_file=output_file
    )
    
    if success:
        print(f"🎉 All done! Your podcast is ready at: {output_file}")
    else:
        print("❌ Podcast generation failed.")


if __name__ == "__main__":
    main()

