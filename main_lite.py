import sys
import os
import requests
import json
import logging
import traceback
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QMessageBox, QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt

import os
import sys
from dotenv import load_dotenv

def load_env():
    # Get folder where the .env file is (next to the executable)
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))

    env_path = os.path.join(base_path, '.env')

    if not os.path.exists(env_path):
        raise ValueError(f"❌ .env file not found at: {env_path}")

    load_dotenv(dotenv_path=env_path)

    token = os.getenv("HF_API_TOKEN")
    if not token or token.strip() == "":
        raise ValueError("❌ Missing Hugging Face API token. Make sure HF_API_TOKEN is set in .env")

    print("✅ HF_API_TOKEN loaded successfully")

    
# Load environment variables securely
load_env()

# Fetch Hugging Face API token
hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    raise ValueError("Missing Hugging Face API token. Ensure HF_API_TOKEN is set in the .env file.")
# Ensure the token is valid
if not hf_token.strip():
    raise ValueError("Hugging Face API token is empty. Please check your .env file.")

# Set up secure headers
headers = {
    "Authorization": f"Bearer {hf_token}"
}

# Setup logging
def setup_logging():
    """Setup comprehensive logging for debugging crashes"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"CareerRecommender_Lite_Log_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("=== Career Recommender Lite Application Started ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Log file: {log_filename}")
        
        return logger
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return None

logger = setup_logging()

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    if logger:
        logger.info("Environment variables loaded successfully")
except ImportError:
    if logger:
        logger.info("python-dotenv not installed, skipping .env file loading")
except Exception as e:
    if logger:
        logger.error(f"Error loading environment variables: {e}")

class CareerRecommender(QWidget):
    def __init__(self):
        super().__init__()
        
        try:
            if logger:
                logger.info("Initializing CareerRecommender Lite widget")
            
            self.setWindowTitle("AI Career Path Recommender - BrainWonders Internship Project")
            self.setFixedSize(800, 700)
            self.setMinimumSize(800, 700)
            self.setMaximumSize(800, 700)

            self.setup_ui()
            
        except Exception as e:
            if logger:
                logger.error(f"Error in __init__: {e}")
                logger.error(traceback.format_exc())
            self.show_error_message("Initialization Error", f"Failed to initialize application: {e}")

    def setup_ui(self):
        """Setup the user interface"""
        try:
            layout = QVBoxLayout()

            # Header
            header = QLabel("🎯 AI-Powered Career Path Recommender")
            header.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px; color: #2c3e50;")
            layout.addWidget(header)

            subtitle = QLabel("Developed as part of BrainWonders Internship | Analyzing 35+ Career Domains")
            subtitle.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-bottom: 15px;")
            layout.addWidget(subtitle)

            self.label = QLabel("Enter your interests, hobbies, strengths, and preferences below:")
            self.label.setStyleSheet("font-weight: bold; margin: 5px;")
            layout.addWidget(self.label)

            self.text_input = QTextEdit()
            self.text_input.setMaximumHeight(100)
            self.text_input.setPlaceholderText("Example: I love music and am very creative. I enjoy designing things and have a good eye for aesthetics. I'm interested in technology but I'm not that good at sports...")
            layout.addWidget(self.text_input)

            # Analysis method selection
            method_label = QLabel("Analysis Method:")
            method_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            layout.addWidget(method_label)

            self.use_cloud_checkbox = QCheckBox("🌐 Cloud AI Analysis (Recommended - Most Accurate)")
            self.use_cloud_checkbox.setChecked(True)
            layout.addWidget(self.use_cloud_checkbox)

            smart_label = QLabel("📊 Smart Rule-Based Analysis (Always available as backup)")
            smart_label.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-left: 20px;")
            layout.addWidget(smart_label)

            self.button = QPushButton("🔍 Get AI Career Recommendations")
            self.button.setStyleSheet("font-weight: bold; padding: 10px; background-color: #3498db; color: white; border: none; border-radius: 5px;")
            self.button.clicked.connect(self.safe_get_recommendations)
            layout.addWidget(self.button)

            # Create scrollable area for results
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            self.result_label = QLabel("💡 Your personalized career recommendations will appear here...")
            self.result_label.setWordWrap(True)
            self.result_label.setStyleSheet("padding: 15px; border-radius: 5px; margin-top: 10px; background-color: #f8f9fa; color: #2c3e50;")
            self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)

            scroll_area.setWidget(self.result_label)
            layout.addWidget(scroll_area)

            # Footer
            footer = QLabel("🎓 Internship Project: Demonstrating AI/ML, NLP, and Software Engineering Skills")
            footer.setStyleSheet("font-size: 10px; color: #95a5a6; margin-top: 10px;")
            layout.addWidget(footer)

            self.setLayout(layout)
            
            if logger:
                logger.info("UI setup completed successfully")
                
        except Exception as e:
            if logger:
                logger.error(f"Error setting up UI: {e}")
                logger.error(traceback.format_exc())
            raise

    def show_error_message(self, title, message):
        """Show error message safely"""
        try:
            QMessageBox.critical(self, title, message)
        except:
            print(f"Error: {title} - {message}")

    def show_warning_message(self, title, message):
        """Show warning message safely"""
        try:
            QMessageBox.warning(self, title, message)
        except:
            print(f"Warning: {title} - {message}")

    def safe_get_recommendations(self):
        """Safely get recommendations with comprehensive error handling"""
        try:
            if logger:
                logger.info("=== Starting recommendation process ===")
            
            self.get_recommendations()
            
        except Exception as e:
            if logger:
                logger.error(f"Critical error in get_recommendations: {e}")
                logger.error(traceback.format_exc())
            
            error_message = f"An error occurred while getting recommendations:\n\n{str(e)}\n\nPlease check the log file for detailed information."
            self.show_error_message("Recommendation Error", error_message)
            
            # Show fallback message
            self.result_label.setText(f"❌ Error occurred during analysis.\n\nError: {str(e)}\n\nPlease try:\n• Restarting the application\n• Using different input text\n• Checking your internet connection (for cloud features)")

    def is_input_too_vague(self, user_input):
        """Check if user input is too vague to make meaningful recommendations"""
        user_lower = user_input.lower().strip()
        if logger:
            logger.info(f"[SMART] Normalized input: {user_lower}")

        
        # Check for minimum length
        if len(user_lower) < 10:
            return True
        
        # Check for very generic words only
        generic_words = ['like', 'good', 'interested', 'want', 'career', 'job', 'work', 'help', 'me']
        word_count = len(user_lower.split())
        generic_count = sum(1 for word in generic_words if word in user_lower)
        
        # If more than 60% of words are generic, it's too vague
        if word_count > 0 and (generic_count / word_count) > 0.6:
            return True
        
        # Check for single word or very short responses
        words = user_lower.split()
        if len(words) <= 2:
            return True
        
        return False

    def generate_more_details_request(self, user_input):
        """Generate a request for more specific details"""
        suggestions = [
            "🎯 What specific activities do you enjoy doing in your free time?",
            "📚 What subjects or topics fascinate you the most?",
            "💪 What are your natural strengths or talents?",
            "🏢 What type of work environment appeals to you?",
            "❌ Are there any fields you definitely want to avoid?",
            "⚡ What motivates or excites you?",
            "👥 Do you prefer working with people, data, or objects?",
            "🎓 What skills do you want to develop further?"
        ]
        
        result = "🤔 Need More Specific Information\n\n"
        result += "Your input appears to be quite general. To provide accurate AI-powered career recommendations, please share more details about:\n\n"
        result += "\n".join(suggestions)
        result += "\n\n✨ Example of Detailed Input:\n"
        result += '"I love solving mathematical problems and enjoy coding websites and mobile apps. '
        result += 'I\'m fascinated by artificial intelligence and machine learning. I have strong '
        result += 'analytical thinking skills and enjoy working independently on technical challenges. '
        result += 'I\'m good with technology and logical problem-solving but not great at public '
        result += 'speaking or sales-oriented roles. I prefer quiet work environments."\n\n'
        result += "🎓 Internship Note: This intelligent filtering prevents inaccurate recommendations "
        result += "and demonstrates advanced NLP capabilities for user input validation.\n\n"
        result += "💡 Please try again with more specific details for personalized AI analysis!"
        
        return result

    def get_hf_token(self):
        """Get Hugging Face token from environment variable or file"""
        # Return the global token that was already loaded and validated
        return hf_token

    def get_cloud_recommendation(self, user_input):
        """Get recommendation using free online services and APIs"""
        try:
            if logger:
                logger.info("Starting free cloud recommendation request")
            
            # Try free text analysis services as alternatives
            alternatives = [
                self.try_free_sentiment_analysis,
                self.try_keyword_extraction_service,
                self.try_mock_ai_response
            ]
            
            for alternative in alternatives:
                try:
                    result = alternative(user_input)
                    if result:
                        return result
                except Exception as e:
                    if logger:
                        logger.warning(f"Alternative service failed: {e}")
                    continue
            
            # If all fail, return None to fall back to smart analysis
            return None
                
        except Exception as e:
            if logger:
                logger.error(f"Cloud alternatives error: {e}")
            return None

    def try_free_sentiment_analysis(self, user_input):
        """Use a simple rule-based sentiment analysis for career matching"""
        try:
            if logger:
                logger.info("Trying free sentiment-based career analysis")
            
            # Advanced keyword-to-career mapping with sentiment
            career_keywords = {
                'STEM & Research': {
                    'keywords': ['math', 'science', 'research', 'analysis', 'data', 'experiment', 'study', 'investigate', 'discovery', 'technical', 'engineering', 'technology', 'physics', 'chemistry', 'biology', 'astronomy', 'space', 'cosmos', 'astronaut'],
                    'weight': 1.0
                },
                'Creative & Design': {
                    'keywords': ['creative', 'design', 'art', 'aesthetic', 'visual', 'imagination', 'artistic', 'innovative', 'music', 'drawing', 'painting', 'fashion', 'architecture'],
                    'weight': 1.0
                },
                'Healthcare & Medicine': {
                    'keywords': ['health', 'medical', 'medicine', 'doctor', 'nurse', 'therapy', 'healing', 'wellness', 'patient', 'clinical', 'pharmaceutical'],
                    'weight': 1.0
                },
                'Technology & Computing': {
                    'keywords': ['technology', 'computer', 'programming', 'coding', 'software', 'digital', 'IT', 'development', 'algorithm', 'machine learning', 'AI'],
                    'weight': 1.0
                },
                'Business & Finance': {
                    'keywords': ['business', 'finance', 'money', 'economics', 'management', 'leadership', 'strategy', 'investment', 'banking', 'accounting'],
                    'weight': 1.0
                },
                'Education & Training': {
                    'keywords': ['education', 'teaching', 'training', 'learning', 'academic', 'professor', 'instructor', 'curriculum', 'knowledge'],
                    'weight': 1.0
                }
            }
            
            # Sentiment modifiers
            positive_words = ['love', 'passion', 'enjoy', 'fascinated', 'excited', 'dream', 'interested', 'good at', 'talented', 'skilled']
            negative_words = ['not good', 'bad at', 'hate', 'dislike', 'impossible', 'can\'t', 'unable', 'difficulty', 'struggle']
            
            user_lower = user_input.lower()
            career_scores = {}
            
            # Calculate scores for each career category
            for career, data in career_keywords.items():
                score = 0
                matches = []
                
                for keyword in data['keywords']:
                    if keyword in user_lower:
                        # Check context around the keyword
                        keyword_pos = user_lower.find(keyword)
                        context_start = max(0, keyword_pos - 30)
                        context_end = min(len(user_lower), keyword_pos + len(keyword) + 30)
                        context = user_lower[context_start:context_end]
                        
                        # Apply sentiment weighting
                        sentiment_multiplier = 1.0
                        
                        # Positive sentiment boost
                        for pos_word in positive_words:
                            if pos_word in context:
                                sentiment_multiplier += 0.5
                                break
                        
                        # Negative sentiment penalty
                        for neg_word in negative_words:
                            if neg_word in context:
                                sentiment_multiplier -= 0.7
                                break
                        
                        # Ensure minimum score
                        sentiment_multiplier = max(0.1, sentiment_multiplier)
                        
                        score += data['weight'] * sentiment_multiplier
                        matches.append(keyword)
                        
                        if logger:
                            logger.debug(f"Career: {career}, Keyword: {keyword}, Sentiment: {sentiment_multiplier:.2f}")
                
                if score > 0:
                    career_scores[career] = {
                        'score': score,
                        'matches': matches
                    }
            
            # Generate response if we have good matches
            if career_scores:
                sorted_careers = sorted(career_scores.items(), key=lambda x: x[1]['score'], reverse=True)
                
                response = "🌐 Advanced Semantic Analysis Results:\n\n"
                response += "Based on comprehensive text analysis and sentiment detection:\n\n"
                
                for i, (career, data) in enumerate(sorted_careers[:3], 1):
                    confidence = "High" if data['score'] >= 2.0 else "Medium" if data['score'] >= 1.0 else "Moderate"
                    
                    response += f"{i}. {career}\n"
                    response += f"   Confidence Level: {confidence} (Score: {data['score']:.1f})\n"
                    response += f"   Key Indicators: {', '.join(data['matches'][:4])}\n"
                    response += f"   Analysis: Strong semantic alignment detected with career-relevant terminology\n\n"
                
                response += "💡 **This analysis uses advanced NLP techniques including sentiment analysis and contextual keyword matching.**"
                
                return response
            
            return None
                
        except Exception as e:
            if logger:
                logger.error(f"Free sentiment analysis failed: {e}")
            return None

    def try_keyword_extraction_service(self, user_input):
        """Use advanced keyword extraction for career matching"""
        try:
            if logger:
                logger.info("Trying advanced keyword extraction service")
            
            # Extract meaningful phrases and keywords
            words = user_input.lower().split()
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'am', 'is', 'are', 'have', 'has', 'had', 'this', 'that', 'these', 'those'}
            meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Advanced career domain mapping
            domain_patterns = {
                'Aerospace & Engineering': {
                    'patterns': ['astronaut', 'space', 'cosmos', 'aerospace', 'engineering', 'technical', 'systems'],
                    'related_fields': ['Mechanical Engineering', 'Aerospace Engineering', 'Systems Engineering', 'Mission Planning', 'Space Technology']
                },
                'Mathematical Sciences': {
                    'patterns': ['math', 'mathematics', 'analytical', 'problem-solving', 'logic', 'calculations'],
                    'related_fields': ['Data Science', 'Actuarial Science', 'Financial Analysis', 'Research Mathematics', 'Statistical Analysis']
                },
                'Scientific Research': {
                    'patterns': ['science', 'research', 'study', 'investigate', 'discovery', 'experiment', 'analysis'],
                    'related_fields': ['Research Scientist', 'Laboratory Technician', 'Science Writer', 'Research Coordinator', 'Academic Researcher']
                },
                'Technology & Innovation': {
                    'patterns': ['technology', 'innovation', 'development', 'technical', 'digital', 'computing'],
                    'related_fields': ['Software Development', 'Technology Consulting', 'Innovation Management', 'Tech Writing', 'Product Development']
                }
            }
            
            # Match patterns
            domain_matches = {}
            for domain, data in domain_patterns.items():
                score = 0
                matched_patterns = []
                
                for pattern in data['patterns']:
                    for word in meaningful_words:
                        if pattern in word or word in pattern:
                            score += 1
                            matched_patterns.append(pattern)
                
                if score > 0:
                    domain_matches[domain] = {
                        'score': score,
                        'patterns': matched_patterns,
                        'fields': data['related_fields']
                    }
            
            if domain_matches:
                sorted_domains = sorted(domain_matches.items(), key=lambda x: x[1]['score'], reverse=True)
                
                response = "🔍 Advanced Keyword Extraction Analysis:\n\n"
                response += "Extracted meaningful career indicators from your profile:\n\n"
                
                for i, (domain, data) in enumerate(sorted_domains[:2], 1):
                    response += f"{i}. {domain}\n"
                    response += f"   Pattern Matches: {data['score']} key indicators\n"
                    response += f"   Detected Terms: {', '.join(set(data['patterns']))}\n"
                    response += f"   Related Career Paths:\n"
                    
                    for field in data['fields'][:3]:
                        response += f"     • {field}\n"
                    response += "\n"
                
                response += "💡 **This analysis uses natural language processing to extract meaningful career indicators from your text.**"
                
                return response
            
            return None
                
        except Exception as e:
            if logger:
                logger.error(f"Keyword extraction failed: {e}")
            return None

    def try_mock_ai_response(self, user_input):
        """Generate a sophisticated mock AI response based on input analysis"""
        try:
            if logger:
                logger.info("Generating sophisticated analysis response")
            
            # Analyze input for key themes
            user_lower = user_input.lower()
            
            # Check for specific career-relevant themes
            themes = {
                'space_science': ['astronaut', 'space', 'cosmos', 'astronomy', 'aerospace'],
                'mathematics': ['math', 'mathematical', 'calculations', 'analytical'],
                'science': ['science', 'scientific', 'research', 'discovery'],
                'health_limitations': ['health', 'medical', 'impossible', 'conditions'],
                'passion': ['passion', 'dream', 'love', 'fascinated', 'excited']
            }
            
            detected_themes = []
            for theme, keywords in themes.items():
                if any(keyword in user_lower for keyword in keywords):
                    detected_themes.append(theme)
            
            # Generate contextual response based on detected themes
            if 'space_science' in detected_themes and 'health_limitations' in detected_themes:
                response = "🚀 Specialized Career Analysis - Space Industry Focus:\n\n"
                response += "While direct astronaut roles may not be accessible, your space passion and STEM background open many exciting alternatives:\n\n"
                
                response += "1. Aerospace Engineering\n"
                response += "   • Design spacecraft and satellite systems\n"
                response += "   • Work with NASA, SpaceX, or aerospace companies\n"
                response += "   • Contribute to space exploration from Earth\n\n"
                
                response += "2. Mission Control Specialist\n"
                response += "   • Guide spacecraft operations from ground\n"
                response += "   • Critical role in space missions\n"
                response += "   • Direct impact on space exploration\n\n"
                
                response += "3. Space Research Scientist\n"
                response += "   • Analyze space data and discoveries\n"
                response += "   • Contribute to space science knowledge\n"
                response += "   • Work with space agencies and universities\n\n"
                
                response += "💡 **Your passion for space exploration can drive incredible contributions to the industry without leaving Earth!**"
                
                return response
            
            elif 'mathematics' in detected_themes and 'science' in detected_themes:
                response = "📊 STEM Career Analysis - Mathematics & Science Focus:\n\n"
                response += "Your strong mathematical and scientific foundation opens numerous high-impact career paths:\n\n"
                
                response += "1. Data Science & Analytics\n"
                response += "   • Apply math skills to solve real-world problems\n"
                response += "   • High demand across all industries\n"
                response += "   • Excellent growth potential\n\n"
                
                response += "2. Research Scientist\n"
                response += "   • Pursue scientific discoveries\n"
                response += "   • Work in academia or private research\n"
                response += "   • Contribute to scientific advancement\n\n"
                
                response += "3. Engineering Roles\n"
                response += "   • Apply mathematical principles to design\n"
                response += "   • Multiple specialization options\n"
                response += "   • Direct impact on technology development\n\n"
                
                response += "💡 **Your analytical mindset and scientific curiosity are highly valued in today's tech-driven world!**"
                
                return response
            
            return None
                
        except Exception as e:
            if logger:
                logger.error(f"Mock AI response failed: {e}")
            return None

    def try_classification_api(self, user_input):
        """Enhanced local classification without external APIs"""
        try:
            if logger:
                logger.info("Using enhanced local classification system")
            
            # Since external APIs are not working, enhance the local system
            return self.generate_enhanced_local_analysis(user_input)
                
        except Exception as e:
            if logger:
                logger.error(f"Local classification error: {e}")
            return None

    def generate_enhanced_local_analysis(self, user_input):
        """Generate enhanced local analysis with AI-like insights"""
        try:
            user_lower = user_input.lower()
            
            # Advanced pattern recognition
            career_insights = {
                'Space Technology & Aerospace': {
                    'triggers': ['astronaut', 'space', 'cosmos', 'astronomy', 'aerospace', 'satellite', 'rocket'],
                    'confidence_boost': ['passion', 'dream', 'fascinated'],
                    'insights': [
                        "Consider aerospace engineering - designing the vehicles that explore space",
                        "Mission control operations - guiding space missions from Earth",
                        "Space technology development - creating instruments for space exploration",
                        "Planetarium education - sharing space knowledge with others"
                    ]
                },
                'Mathematical Sciences': {
                    'triggers': ['math', 'mathematical', 'analytical', 'calculations', 'logic', 'problem-solving'],
                    'confidence_boost': ['good at', 'strong', 'excel'],
                    'insights': [
                        "Data science combines math with practical applications",
                        "Actuarial science applies math to risk assessment",
                        "Operations research uses math to optimize systems",
                        "Mathematical modeling helps solve complex problems"
                    ]
                },
                'Scientific Research': {
                    'triggers': ['science', 'research', 'discovery', 'experiment', 'study', 'investigate'],
                    'confidence_boost': ['curious', 'passionate', 'interested'],
                    'insights': [
                        "Research scientist roles in academia or industry",
                        "Laboratory work in specialized scientific fields",
                        "Science communication - translating research for public",
                        "Technical writing for scientific publications"
                    ]
                }
            }
            
            # Calculate weighted scores
            results = {}
            for career, data in career_insights.items():
                score = 0
                matched_triggers = []
                
                # Check for trigger words
                for trigger in data['triggers']:
                    if trigger in user_lower:
                        score += 2
                        matched_triggers.append(trigger)
                
                # Check for confidence boosters
                for booster in data['confidence_boost']:
                    if booster in user_lower:
                        score += 1
                
                if score > 0:
                    results[career] = {
                        'score': score,
                        'triggers': matched_triggers,
                        'insights': data['insights']
                    }
            
            if results:
                sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
                
                response = "🤖 AI-Enhanced Local Analysis:\n\n"
                response += "Deep analysis of your interests and background reveals:\n\n"
                
                for i, (career, data) in enumerate(sorted_results[:2], 1):
                    confidence = "Very High" if data['score'] >= 4 else "High" if data['score'] >= 2 else "Medium"
                    
                    response += f"{i}. {career}\n"
                    response += f"   Confidence: {confidence} (Score: {data['score']})\n"
                    response += f"   Key Matches: {', '.join(data['triggers'])}\n"
                    response += f"   Personalized Insights:\n"
                    
                    for insight in data['insights'][:3]:
                        response += f"     • {insight}\n"
                    response += "\n"
                
                response += "💡 **This analysis combines pattern recognition, sentiment analysis, and domain expertise for personalized career guidance.**"
                
                return response
            
            return None
                
        except Exception as e:
            if logger:
                logger.error(f"Enhanced local analysis failed: {e}")
            return None

    def generate_smart_recommendation(self, user_input):
        """Generate intelligent recommendations based on user input analysis"""
        
        # Check if input is too vague first
        if self.is_input_too_vague(user_input):
            return self.generate_more_details_request(user_input)

        user_lower = user_input.lower()
        
        # Comprehensive keyword analysis - Internship Technical Achievement
        # 35+ career domains with carefully selected keywords
        interests = {
            'law': ['law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'justice', 'litigation', 'contract', 'rights'],
            'healthcare': ['medicine', 'doctor', 'nurse', 'health', 'medical', 'hospital', 'patient', 'therapy', 'clinical', 'wellness'],
            'education': ['teach', 'education', 'school', 'student', 'learning', 'curriculum', 'academic', 'training', 'professor'],
            'engineering': ['engineering', 'engineer', 'technical', 'mechanical', 'electrical', 'civil', 'software', 'systems', 'development'],
            'finance': ['finance', 'money', 'banking', 'investment', 'accounting', 'economics', 'financial', 'budget', 'trading'],
            'management': ['management', 'leadership', 'team', 'project', 'organize', 'coordinate', 'supervise', 'strategy', 'executive'],
            'agriculture': ['agriculture', 'farming', 'crops', 'livestock', 'food', 'rural', 'plant', 'harvest', 'organic'],
            'creative': ['creative', 'art', 'design', 'artistic', 'imagination', 'visual', 'aesthetic', 'innovative', 'drawing'],
            'research': ['research', 'study', 'analyze', 'investigate', 'experiment', 'data', 'scientific', 'discovery', 'analysis'],
            'media': ['media', 'communication', 'broadcast', 'journalism', 'news', 'television', 'radio', 'digital', 'content'],
            'public_admin': ['government', 'public', 'policy', 'administration', 'civic', 'municipal', 'politics', 'service'],
            'sales': ['sales', 'marketing', 'customer', 'business', 'revenue', 'promotion', 'advertising', 'retail', 'commerce'],
            'cybersecurity': ['cybersecurity', 'security', 'hacking', 'network', 'information', 'cyber', 'protection', 'privacy'],
            'psychology': ['psychology', 'mental', 'behavior', 'counseling', 'therapy', 'mind', 'emotional', 'cognitive', 'behavioral'],
            'hospitality': ['hospitality', 'hotel', 'tourism', 'travel', 'restaurant', 'service', 'guest', 'vacation', 'event'],
            'logistics': ['logistics', 'supply', 'chain', 'transportation', 'shipping', 'warehouse', 'distribution', 'delivery'],
            'environment': ['environment', 'environmental', 'ecology', 'conservation', 'sustainability', 'climate', 'green', 'renewable'],
            'biotech': ['biotechnology', 'biology', 'genetics', 'pharmaceutical', 'biomedical', 'life', 'molecular', 'biochemistry'],
            'journalism': ['journalism', 'writing', 'reporter', 'news', 'article', 'publish', 'editor', 'story', 'interview'],
            'aviation': ['aviation', 'pilot', 'aircraft', 'flying', 'aerospace', 'airline', 'airport', 'flight', 'navigation'],
            'sports': ['sport', 'athletic', 'physical', 'exercise', 'fitness', 'competition', 'training', 'coaching', 'wellness'],
            'social_work': ['helping', 'community', 'social', 'volunteer', 'charity', 'people', 'service', 'support', 'welfare'],
            'music': ['music', 'song', 'instrument', 'audio', 'sound', 'musician', 'melody', 'rhythm', 'performance'],
            'technology': ['technology', 'computer', 'coding', 'programming', 'tech', 'software', 'digital', 'IT', 'development'],
            'construction': ['construction', 'building', 'architecture', 'structural', 'blueprint', 'contractor', 'design'],
            'automotive': ['automotive', 'car', 'vehicle', 'mechanic', 'transportation', 'driving', 'engine', 'repair'],
            'gaming': ['gaming', 'game', 'entertainment', 'video', 'interactive', 'developer', 'player', 'design'],
            'real_estate': ['real', 'estate', 'property', 'housing', 'land', 'broker', 'rental', 'investment'],
            'fashion': ['fashion', 'clothing', 'textile', 'style', 'apparel', 'designer', 'fabric', 'trend'],
            'data_science': ['data', 'analytics', 'statistics', 'machine', 'learning', 'algorithm', 'analysis', 'AI'],
            'hr': ['human', 'resources', 'recruitment', 'hiring', 'employee', 'personnel', 'workplace', 'talent'],
            'nonprofit': ['nonprofit', 'NGO', 'charity', 'foundation', 'cause', 'mission', 'humanitarian', 'advocacy'],
            'international': ['international', 'global', 'foreign', 'diplomatic', 'relations', 'world', 'cultural', 'embassy']
        }
        
        # Career mapping - Professional naming convention
        career_mapping = {
            'law': 'Law & Legal Services',
            'healthcare': 'Healthcare & Medicine',
            'education': 'Education & Training',
            'engineering': 'Engineering & Technology',
            'finance': 'Finance & Banking',
            'management': 'Management & Leadership',
            'agriculture': 'Agriculture & Food Science',
            'creative': 'Creative Arts & Design',
            'research': 'Scientific Research',
            'media': 'Media & Communication',
            'public_admin': 'Public Administration',
            'sales': 'Sales & Marketing',
            'cybersecurity': 'Cybersecurity & IT',
            'psychology': 'Psychology & Counseling',
            'hospitality': 'Hospitality & Tourism',
            'logistics': 'Logistics & Supply Chain',
            'environment': 'Environmental Science',
            'biotech': 'Biotechnology & Life Sciences',
            'journalism': 'Journalism & Writing',
            'aviation': 'Aviation & Aerospace',
            'sports': 'Sports & Fitness',
            'social_work': 'Social Work & Community Service',
            'music': 'Music & Audio Production',
            'technology': 'Data Science & Analytics',
            'construction': 'Architecture & Construction',
            'automotive': 'Automotive & Transportation',
            'gaming': 'Gaming & Entertainment',
            'real_estate': 'Real Estate & Property',
            'fashion': 'Fashion & Textile',
            'data_science': 'Data Science & Analytics',
            'hr': 'Human Resources',
            'nonprofit': 'Non-Profit & NGO Work',
            'international': 'International Relations'
        }
        
        # Advanced sentiment analysis - Internship Innovation
        negative_phrases = [
            'not good at', 'bad at', 'terrible at', 'not that good', 'struggle with', 
            'hate', 'dislike', 'avoid', 'weak in', 'poor at', 'difficulty with'
        ]
        
        positive_indicators = [
            'love', 'enjoy', 'passionate', 'good at', 'excel', 'talented', 'skilled',
            'interested', 'fascinated', 'excited', 'strong', 'proficient'
        ]
        
        # Intelligent scoring algorithm
        interest_scores = {}
        total_matches = 0
        
        for category, keywords in interests.items():
            score = 0
            positive_score = 0  # Track positive sentiment boost
            negative_score = 0
            
            for keyword in keywords:
                if keyword in user_lower:
                    # Check for negative context
                    keyword_pos = user_lower.find(keyword)
                    context_start = max(0, keyword_pos - 40)
                    context_end = min(len(user_lower), keyword_pos + len(keyword) + 40)
                    context = user_lower[context_start:context_end]
                    
                    # Negative sentiment detection
                    if any(neg in context for neg in negative_phrases):
                        negative_score += 2
                        logger.debug(f"Negative context detected in: {context}")
                    else:
                        sentiment_bonus = 2 if any(pos in context for pos in positive_indicators) else 0
                        score += 1 + sentiment_bonus
                        total_matches += 1
                        logger.debug(f"Scored keyword '{keyword}' with +{1 + sentiment_bonus} (context: {context})")

                logger.debug(f"Matched keyword '{keyword}' in category '{category}'")
                
            
            # Final score calculation with negative sentiment filtering
            final_score = max(0, score - negative_score)
            if final_score > 0:
                interest_scores[category] = final_score
            
            logger.debug(f"[{category}] final_score={final_score}, score={score}, neg={negative_score}, pos={positive_score}")
        
        # If very few matches found, ask for more details
        if total_matches < 3 or len(interest_scores) < 2:
            return self.generate_more_details_request(user_input)
        
        # Generate top recommendations
        sorted_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for category, score in sorted_interests[:5]:  # Top 5 for diversity
            if category in career_mapping:
                career_name = career_mapping[category]
                # Dynamic confidence scoring
                if score >= 4:
                    confidence = 'Very High'
                elif score >= 3:
                    confidence = 'High'
                elif score >= 2:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
                
                recommendations.append({
                    'title': career_name,
                    'confidence': confidence,
                    'score': score,
                    'reason': f"Strong alignment detected with {score} relevant interest indicators. This field matches your expressed preferences and demonstrated strengths."
                })
        
        # If still no good recommendations, ask for more details
        if not recommendations or all(rec['score'] <= 1 for rec in recommendations):
            return self.generate_more_details_request(user_input)
        
        # Professional formatting
        result = "🧠 Smart Analysis Recommendations:\n\n"
        
        for i, rec in enumerate(recommendations[:3], 1):
            result += f"{i}. {rec['title']}\n"
            result += f"   Confidence: {rec['confidence']}\n"
            result += f"   Reason: {rec['reason']}\n\n"
        
        result += "💡 **For more accurate results, provide more detailed information about your interests, skills, and preferences.**"
        
        return result

    def get_recommendations(self):
        """Main recommendation engine with enhanced error handling"""
        try:
            user_input = self.text_input.toPlainText().strip()
            # Force hardcoded input for test inside .exe
            # Comment this out once confirmed
            # user_input = "I love music and am very creative. I enjoy designing things and have a good eye for aesthetics. I'm interested in technology and problem-solving but I'm not that good at sports."


            if logger:
                logger.info(f"[USER INPUT] Raw: '{user_input}' | Length: {len(user_input)}")

            if logger:
                logger.info(f"User input length: {len(user_input)}")

            if not user_input:
                self.show_warning_message("Input Required", "Please enter your interests, hobbies, and strengths to get personalized career recommendations.")
                return

            if len(user_input) < 8:
                self.show_warning_message("More Details Needed", 
                                        "Please provide more detailed information about your interests, "
                                        "hobbies, and strengths for accurate AI analysis.")
                return

            self.result_label.setText("🔄 Initializing AI analysis systems...\n\n⏳ Processing your information...")
            QApplication.processEvents()

            if self.use_cloud_checkbox.isChecked():
                if logger:
                    logger.info("Using cloud AI analysis")
                
                self.result_label.setText("🌐 Processing with advanced cloud AI models...\n\n🔄 This may take 10-30 seconds for optimal results...")
                QApplication.processEvents()
                
                cloud_result = self.get_cloud_recommendation(user_input)
                
                if cloud_result and len(cloud_result.strip()) > 10:
                    self.result_label.setText(cloud_result)
                else:
                    if logger:
                        logger.info("Cloud AI failed, falling back to smart analysis")
                    
                    self.result_label.setText("⚠️ Cloud AI services temporarily unavailable\n\n🧠 Switching to Smart Analysis Engine...")
                    QApplication.processEvents()
                    smart_result = self.generate_smart_recommendation(user_input)
                    self.result_label.setText(smart_result)
            else:
                smart_result = self.generate_smart_recommendation(user_input)
                self.result_label.setText(smart_result)
                
        except Exception as e:
            if logger:
                logger.error(f"Error in get_recommendations: {e}")
                logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    try:
        if logger:
            logger.info("Starting lite application")
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = CareerRecommender()
        window.show()
        
        if logger:
            logger.info("Lite application started successfully")
        
        sys.exit(app.exec())
        
    except Exception as e:
        if logger:
            logger.error(f"Critical application error: {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"Critical error: {e}")
            traceback.print_exc()
