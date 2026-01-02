# SyntheticRadioHost
An AI-powered podcast generation system that creates natural, conversational Hinglish (Hindi-English) radio shows from any Wikipedia topic. Combines Large Language Models (Ollama) with advanced Text-to-Speech (ElevenLabs) to produce engaging, multi-voice podcast content automatically. 

The system leverages cutting-edge AI technologies:
• Ollama (Local LLM) for generating authentic Hinglish dialogue with natural code-switching
• ElevenLabs API for high-quality, emotion-aware text-to-speech synthesis
• Wikipedia API for fetching contextual information

## ✨ Features

- **🌐 Wikipedia Integration**: Automatically fetches contextual information from Wikipedia for any topic
- **🤖 LLM-Powered Scripts**: Uses Ollama to generate natural, conversational Hinglish dialogue
- **🎭 Multi-Voice Podcasts**: Creates dynamic conversations between two hosts (Vineet & Simran) with distinct personalities
- **🎵 High-Quality Audio**: Leverages ElevenLabs API for realistic, emotion-aware text-to-speech
- **💬 Natural Hinglish**: Generates authentic code-switching patterns with phonetic spelling for natural pronunciation
- **🎬 Emotional Expressions**: Supports emotional tags like `[laughs]`, `[sighs]`, `[giggles]` for realistic audio
- **✅ Comprehensive Testing**: 45+ unit tests ensuring reliability and robustness
- **📊 Progress Tracking**: Real-time logging and progress indicators throughout the generation process

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Wikipedia  │────▶│  Ollama LLM  │────▶│   Script    │────▶│ ElevenLabs   │
│   Context   │     │   (llama3.2) │     │   Parser    │     │   TTS API    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                    │
                                                                    ▼
                                                            ┌──────────────┐
                                                            │   Final MP3  │
                                                            │   Podcast    │
                                                            └──────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **Ollama** installed and running locally ([Download Ollama](https://ollama.ai/))
- **ElevenLabs API Key** ([Get API Key](https://elevenlabs.io/))
- **Internet Connection** (for Wikipedia and API calls)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/synthetic-radio-host.git
cd synthetic-radio-host/RADIO-HOST
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Ollama

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai/ for installation instructions

# Pull the required model
ollama pull llama3.2

# Start Ollama service (usually runs automatically)
ollama serve
```

### 4. Configure API Keys

#### Option 1: Command-Line Argument (Recommended)
```bash
python syntheticRadioHostScript.py <your_elevenlabs_api_key> <output_file_path>
```

#### Option 2: Environment Variable
Create a `.env` file in your project root:
```env
ELLEVENLABS_API_KEY=your_api_key_here
```

Then run:
```bash
python syntheticRadioHostScript.py
```

### 5. Run the Script

```bash
python syntheticRadioHostScript.py <api_key> output/podcast.mp3
```

The script will:
1. ✅ Check Ollama connection
2. 📚 Fetch Wikipedia context for the topic (default: "Mumbai Indians")
3. 🤖 Generate a Hinglish conversation script
4. 📝 Parse the script into dialogue segments
5. 🎵 Ask for confirmation before generating audio
6. 🎙️ Generate and combine audio segments into final MP3

## 📦 Installation Details

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Test Dependencies (Optional)

```bash
pip install -r requirements-test.txt
```

## 🔧 Configuration

### Modify Topic

Edit the `topic` variable in `main()` function:

```python
def main():
    topic = "Your Topic Here"  # Change this
    # ... rest of the code
```

### Change Voice Models

Edit the `VOICES` dictionary to use different ElevenLabs voices:

```python
VOICES = {
    "Vikram": "hczKB0VbXLcBTn17ShYS",
    "Reyanshi": "UrB5rVw5j9MDZWDZJtOJ",
    "Mihir": "fZpEbMnaJhQJwDY7lFcw",
    "Nikita": "SZfY4K69FwXus87eayHK",
    "Aura": "QbPozfpTbDrP62pzQpyg"
}
```

### Adjust Script Parameters
Modify the prompt in `create_hinglish_prompt()` to change:
- Conversation duration
- Number of dialogue lines
- Word count
- Style and tone

## 🧪 Testing
Run the comprehensive test suite:

```bash
# Run all tests
pytest test_syntheticRadioHostScript.py -v -s
```

### Test Coverage

The project includes **51 comprehensive test cases** covering:
- ✅ Wikipedia context fetching
- ✅ Prompt generation
- ✅ Ollama connection checking
- ✅ Script generation
- ✅ Dialogue parsing
- ✅ Audio segment generation
- ✅ Output path validation
- ✅ Full podcast generation
- ✅ Integration tests

## 📁 Project Structure

```
RADIO-HOST/
├── syntheticRadioHostScript.py    # Main script
├── test_syntheticRadioHostScript.py # Test suite (45 tests)
├── README.md                       # This file
├── .env.example                    # Environment variables template
├── requirements.txt                # Main dependencies
├── requirements-test.txt           # Test dependencies
│
├── Documentation/
│   ├── Hinglish_Prompt_Explanation.txt
│   ├── DESIGN_DOCUMENATION.md
│   ├── Test_Cases_Document
|   └── Technical_Design_Document_SyntheticRadioHost
```

## 🔑 API Keys & Environment Variables

### ElevenLabs API Key
1. Sign up at [ElevenLabs](https://elevenlabs.io/)
2. Get your API key from your account dashboard
3. Use it via command-line argument or `.env` file

### Environment Variables
Create a `.env` file (see `.env.example`):

```env
ELLEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

## 🎨 How It Works

### 1. Context Fetching
The script fetches Wikipedia summaries for the given topic, providing factual context for the conversation.

### 2. Script Generation
Ollama LLM generates a natural Hinglish conversation following specific guidelines:
- Phonetic spelling for Hindi words
- Code-switching patterns (English + Hindi)
- Emotional expressions and fillers
- Short, punchy sentences for natural flow

### 3. Dialogue Parsing
The generated script is parsed into structured dialogue segments, identifying speakers and their lines.

### 4. Audio Generation
Each dialogue segment is converted to speech using ElevenLabs API with:
- Distinct voices for each host
- Emotion-aware synthesis
- Natural pauses and intonations

### 5. Audio Combination
All segments are combined into a single MP3 file, creating the final podcast.

## 🛠️ Technology Stack

- **Python 3.8+**: Core language
- **Ollama**: Local LLM inference (llama3.2 model)
- **ElevenLabs API**: Text-to-Speech synthesis
- **Wikipedia API**: Content fetching
- **pytest**: Testing framework
- **python-dotenv**: Environment variable management

## 📝 Example Output
The script generates a 2-minute conversation like this:

```
Vineet: [pause] Swagat hai doston! Aaj Mumbai Indians ke baare main baat karte hain. 
        IPL champions, hai na? [laughs]

Simran: Oh my god, literally... they're so amazing! But tell me, Vineet, 
        kaise they became so successful?

Vineet: Arre, it's all about the team spirit, Simran. Look at Rohit Sharma... 
        bilkul bindaas captain hai na!
```

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve
```

### Wikipedia API Errors
- Check your internet connection
- Verify the topic exists on Wikipedia
- Some topics may have limited content

### ElevenLabs API Errors
- Verify your API key is correct
- Check your API quota/credits
- Ensure you have sufficient characters remaining

### Audio Generation Fails
- Check API key validity
- Verify voice IDs are correct
- Ensure output directory has write permissions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
---

**Made with ❤️ for the AI/LLM Integration community**
