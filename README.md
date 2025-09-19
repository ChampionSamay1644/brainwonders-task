# AI-Powered Career Path Recommender

## 🎯 Project Overview

This is an **AI-powered career recommendation system** developed as part of an internship project at **BrainWonders**. The application leverages advanced natural language processing and sentiment analysis to analyze user profiles and provide personalized career guidance.

## 🚀 Quick Start

### Prerequisites

- Python 3.7+ installed on your system
- Windows, macOS, or Linux

### Installation & Running

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ChampionSamay1644/brainwonders-task.git
   cd brainwonders-task
   ```

2. **Create and activate virtual environment**:

   ```bash
   python -m venv career_env

   # On Windows:
   career_env\Scripts\activate

   # On macOS/Linux:
   source career_env/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main_lite.py
   ```

## 🧠 What This Application Does

The **Career Path Recommender** is an intelligent desktop application that:

- **Analyzes Personal Profiles**: Takes user input about interests, hobbies, strengths, and preferences
- **Provides AI-Driven Recommendations**: Uses advanced sentiment analysis to suggest the most suitable career paths
- **Offers Detailed Reasoning**: Explains why each career is recommended based on the user's profile
- **Covers 60+ Career Domains**: From traditional fields like Healthcare and Engineering to emerging areas like Cybersecurity and Data Science
- **Advanced Negation Detection**: Intelligently excludes careers that users explicitly reject using sophisticated sentiment analysis

## 🔧 Technical Features

### Advanced Sentiment Analysis Engine

- **Complex Negation Detection**: Handles phrases like "wouldn't be caught coding even if my life depended on it"
- **Clause Boundary Detection**: Prevents negation from crossing grammatical boundaries (e.g., "but", "however")
- **Word Boundary Matching**: Uses regex patterns to avoid false positives from substring matches
- **Contextual Understanding**: Analyzes sentiment within proper linguistic scope

### Enhanced Pattern Recognition

- **60+ Career Domain Coverage**: Comprehensive keyword mapping across diverse professional fields
- **Multi-layered Confidence Scoring**: Based on sentiment strength, keyword frequency, and pattern diversity
- **Sarcasm and Irony Detection**: Identifies indirect negative expressions
- **Professional Output Formatting**: Clean, structured recommendations with reasoning

## 🎯 Example Usage

**Input**: "I wouldn't be caught coding even if my life depended on it but I love art and music and all the creative thinking stuff along with sports and athletics"

**Output**: Recommends Graphic Design & Visual Arts, Fitness & Personal Training, Music Production, etc. (correctly avoids Software Development)

## 📁 Project Structure

```
brainwonders-task/
├── main_lite.py           # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
└── career_env/          # Virtual environment (created after setup)
```

## 🔬 Technical Implementation Details

### Core Algorithm Features

1. **Enhanced Text Preprocessing**

   - Contraction expansion ("can't" → "cannot")
   - Punctuation normalization
   - Whitespace standardization

2. **Advanced Sentiment Analysis**

   - Clause boundary detection using contrasting conjunctions
   - Word boundary regex matching to prevent false positives
   - Graduated sentiment scoring (-15 to +2 scale)
   - Context-aware negation scope analysis

3. **Intelligent Filtering**
   - Minimum input validation to prevent vague responses
   - Quality threshold filtering for recommendations
   - Diversity scoring based on keyword match breadth

### Key Technical Innovations

- **Complex Negation Handling**: Detects and properly scopes strong negative expressions
- **Clause-Aware Processing**: Prevents sentiment from crossing grammatical boundaries
- **Multi-Pattern Recognition**: Combines direct keyword matching with contextual sentiment
- **Professional Output Generation**: Structured recommendations with confidence levels and reasoning

## 🧪 Testing & Validation

The system has been thoroughly tested with edge cases including:

- Strong negative expressions with complex grammar
- Mixed sentiment with contrasting conjunctions
- Substring false positives (e.g., "port" in "sports")
- Sarcasm and indirect rejection patterns

## 🛠️ Development Notes

This project demonstrates:

- Advanced NLP techniques for career counseling
- Robust error handling and graceful degradation
- Professional GUI development with PyQt6
- Comprehensive logging and debugging capabilities
- Clean, maintainable code architecture

## 📄 License

This project is developed as part of an educational internship at BrainWonders and is intended for demonstration purposes.

---

**Developed by**: Champion Samay  
**Project**: BrainWonders Internship  
**Technology Stack**: Python, PyQt6, Advanced NLP, Sentiment Analysis
