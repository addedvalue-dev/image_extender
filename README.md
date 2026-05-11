# Image Extender v0.1.1 - Local Version

A Python application that extends images with sound design capabilities using AI-powered object detection and sound synthesis.

## Overview

Image Extender is a sophisticated tool that can both analyze existing images and generate new ones using AI, then automatically creates immersive soundscapes by identifying objects and scenes, searching, downloading, and mixing appropriate sound files. The application uses computer vision for object detection, AI for image generation and semantic analysis, and advanced audio processing techniques to create realistic audio environments.

## Features

- **AI-Powered Object Detection**: Uses MediaPipe to identify objects, scenes, and locations in images
- **AI Image Generation**: Create realistic images from text prompts using OpenAI's GPT models with automatic prompt enrichment
- **Intelligent Sound Matching**: Leverages OpenAI GPT for semantic sound selection and FreeSound API for audio resources
- **Advanced Audio Processing**: 
  - Professional audio mixing and panning
  - Reverb and spatial audio effects
  - Automatic loudness normalization
  - Background music integration with smart volume control
- **Interactive GUI**: User-friendly tkinter interface with tabbed navigation
- **Real-time Audio Playback**: Preview soundscapes with background music support
- **Export Capabilities**: Save your created soundscapes as audio files

## Requirements

### System Requirements
- Python 3.8 or higher
- Windows, macOS, or Linux
- Minimum 4GB RAM (8GB recommended)
- Audio output device for playback

### Python Dependencies

See `requirements.txt` for the complete list of dependencies. Key libraries include:

- **Computer Vision**: `opencv-python`, `mediapipe`
- **Audio Processing**: `pydub`, `librosa`, `soundfile`, `pedalboard`, `pyloudnorm`
- **AI/ML**: `openai`, `numpy`, `scipy`
- **GUI**: `tkinter` (included with Python)
- **Networking**: `requests`
- **Audio I/O**: `sounddevice`

## Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd image_extender
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare background music (optional)**
   - Place a background music file named `background_music.mp3` in the main directory
   - Supported formats: MP3, WAV, FLAC, OGG

## API Keys Setup

The application requires two API keys for full functionality:

### 1. OpenAI API Key
- Required for AI-powered sound matching and semantic analysis
- Get your key at: https://platform.openai.com/api-keys
- The app will prompt for the key on first launch, or you can set it via File → Set OpenAI API Key

### 2. FreeSound API Key
- Required for downloading sound files
- Get your key at: https://freesound.org/apiv2/apply/
- The app will prompt for the key on first launch, or you can set it via File → Set FreeSound API Key

## AI Image Generation

The application includes powerful AI image generation capabilities:

### How It Works
1. **Prompt Input**: Enter a text description of the image you want to create
2. **Prompt Enrichment**: OpenAI GPT automatically enhances your prompt for realistic photo generation
3. **Image Generation**: Uses OpenAI's image generation models to create high-quality, realistic images
4. **Automatic Integration**: Generated images are immediately available for object detection and soundscape creation

### Features
- **Realistic Photo Style**: Images are optimized to look like real photographs, not illustrations or cartoons
- **Prompt Enhancement**: AI automatically refines your text prompt for better results
- **High Resolution**: Generates 1024x1024 pixel images
- **Seamless Integration**: Generated images work exactly like uploaded images for all features

### Usage Tips
- Be descriptive but concise in your prompts
- Include details about lighting, setting, and composition
- Examples: "A peaceful forest with sunlight filtering through trees", "A busy city street at night with neon lights", "A cozy coffee shop interior with warm lighting"

### Requirements
- OpenAI API key with image generation capabilities
- Internet connection for API calls
- Sufficient API credits (image generation consumes more credits than text processing)

## Usage

1. **Launch the application**
   ```bash
   python image_extender.py
   ```

2. **Choose Image Source**
   - Use the "Image" tab
   - **Option A: Upload Image**
     - Select "Upload Image" radio button
     - Click "Browse" to select an image file
     - Supported formats: JPEG, PNG, BMP, TIFF
   - **Option B: Generate Image**
     - Select "Generate Image" radio button
     - Enter a text description of the image you want to create
     - Click "Generate Image" to create a realistic image using AI

### Image Upload Interface
![Image Upload Tab](product_pics/Screenshot%202026-01-31%20183220.png)

3. **Analyze the image**
   - Click "Analyze Image" to detect objects and scenes
   - Review the detected tags and importance values

### Object Detection Interface
![Object Detection Tab](product_pics/Screenshot%202026-01-31%20183227.png)

4. **Configure sound settings**
   - Switch to the "Sound Settings" tab
   - Adjust parameters like:
     - Duration range
     - Audio quality
     - License preferences
     - File format

5. **Create soundscape**
   - Go to the "Sound Creation" tab
   - Click "Download Sounds and Create Mix"
   - Wait for the process to complete (this may take several minutes)

### Sound Creation Interface
![Sound Creation Tab](product_pics/Screenshot%202026-01-31%20183234.png)

6. **Preview and export**
   - Use playback controls to preview your soundscape
   - Export the final mix using the export options

### Feedback Interface
![Feedback Tab](product_pics/Screenshot%202026-01-31%20190249.png)

## Directory Structure

```
image_extender/
├── image_extender.py          # Main application file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore                # Git ignore file
├── background_music.mp3       # Optional background music
├── downloaded_sounds/         # Downloaded sound files
├── exports/                  # Exported audio files
├── logs/                     # Application logs
├── temp/                     # Temporary files
└── images/                   # Image uploads
```

## Configuration

### Sound Settings
- **Duration**: Set minimum and maximum duration for individual sounds
- **Quality**: Choose between high quality (longer processing) or fast processing
- **License**: Filter sounds by license type (Creative Commons, etc.)
- **File Format**: Preferred audio format for downloads

### Audio Processing
- **Reverb**: Automatic room detection and reverb application
- **Panning**: Intelligent stereo positioning based on image analysis
- **Loudness**: Automatic normalization to industry standards
- **Background Music**: Smart volume control that fades during sound creation

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all packages from `requirements.txt` are installed
   - Some audio libraries may require additional system packages

2. **API Key Errors**
   - Verify your OpenAI and FreeSound API keys are valid
   - Check your internet connection
   - Ensure API keys have sufficient credits/quotas

3. **Audio Playback Issues**
   - Check your audio output device
   - Ensure no other application is blocking the audio device
   - Try restarting the application

4. **Memory Issues**
   - Processing large sound libraries can be memory-intensive
   - Close other applications if experiencing slowdowns
   - Consider reducing quality settings for faster processing

### Performance Tips

- Use SSD storage for faster file operations
- Ensure stable internet connection for API calls
- Process smaller images for faster object detection
- Use high-quality mode only when necessary

## API Usage Limits

- **OpenAI API**: Rate limits apply based on your subscription tier
- **FreeSound API**: Limited requests per day for free accounts
- The application includes intelligent caching to minimize API calls

## Contributing

This is a local adaptation of a Google Colab notebook. For bug reports and feature requests:

1. Check existing issues first
2. Provide detailed error messages and system information
3. Include sample images that reproduce the issue (if applicable)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MediaPipe** for computer vision capabilities
- **OpenAI** for AI-powered semantic analysis
- **FreeSound** for the sound library database
- **Pedalboard** for professional audio processing
- Original Google Colab notebook authors

## Version History

### v0.1.1 (Local Version)
- Local Python adaptation with tkinter GUI
- Added background music support
- Improved audio processing and reverb capabilities
- Enhanced error handling and user feedback
- API key management through GUI

---

**Note**: This application requires internet connectivity for API calls and sound downloads. All downloaded sounds are subject to their respective license terms from FreeSound.
