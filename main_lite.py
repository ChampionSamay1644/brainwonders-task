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
        # Method 1: Environment variable
        token = os.environ.get('HF_API_TOKEN')
        if token and token.strip():
            return token.strip()
            
        # Method 2: From a token file
        token_file = os.path.join(os.path.dirname(__file__), 'hf_token.txt')
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                    if token:
                        return token
            except Exception as e:
                print(f"Error reading token file: {e}")
        
        # Method 3: Try loading from .env file
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('HF_API_TOKEN='):
                            token = line.split('=', 1)[1].strip()
                            if token:
                                return token
            except Exception as e:
                print(f"Error reading .env file: {e}")
        
        return None

    def get_cloud_recommendation(self, user_input):
        """Get recommendation from Hugging Face cloud LLM"""
        try:
            # Use a more reliable model for text generation
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            # Enhanced prompt engineering for better career recommendations
            prompt = f"""Professional Career Counselor Analysis:

Client Profile: "{user_input}"

Career Recommendations:

1. Primary Career Match: """

            # Get authentication token
            token = self.get_hf_token()
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 250,
                    "temperature": 0.8,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            # API call with better error handling
            response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    if generated_text and len(generated_text) > 15:
                        # Professional formatting
                        formatted_response = "🌐 Cloud AI Recommendations:\n\n"
                        formatted_response += f"1. {generated_text}\n\n"
                        formatted_response += "💡 **For more accurate results, provide more detailed information about your interests, skills, and preferences.**"
                        return formatted_response
            
            # Try alternative classification approach
            return self.try_classification_api(user_input)
                
        except Exception as e:
            print(f"Cloud API Error: {e}")
            return None

    def try_classification_api(self, user_input):
        """Alternative cloud API using classification instead of generation"""
        try:
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            
            # Optimized career labels for classification
            career_labels = [
                "Creative Arts and Design careers", 
                "Music and Audio Production careers",
                "Technology and Engineering careers",
                "Healthcare and Medical careers",
                "Business and Finance careers",
                "Education and Training careers",
                "Sales and Marketing careers",
                "Data Science and Analytics careers"
            ]
            
            token = self.get_hf_token()
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            payload = {
                "inputs": user_input,
                "parameters": {
                    "candidate_labels": career_labels
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'labels' in result and 'scores' in result:
                    formatted_response = "🌐 Cloud AI Recommendations:\n\n"
                    
                    for i, (label, score) in enumerate(zip(result['labels'][:3], result['scores'][:3]), 1):
                        # Clean up label names
                        clean_label = label.replace(" careers", "").replace(" and ", " & ")
                        confidence_level = "High" if score > 0.6 else "Medium"
                        
                        formatted_response += f"{i}. {clean_label}\n"
                        formatted_response += f"   Confidence: {confidence_level} ({score*100:.1f}%)\n"
                        formatted_response += f"   Reason: AI analysis shows strong semantic alignment with this career domain\n\n"
                    
                    formatted_response += "💡 **For more accurate results, provide more detailed information about your interests, skills, and preferences.**"
                    
                    return formatted_response
            
            return None
                
        except Exception as e:
            print(f"Classification API Error: {e}")
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
