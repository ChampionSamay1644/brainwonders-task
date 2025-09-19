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

# Setup logging before any other imports
def setup_logging():
    """Setup comprehensive logging for debugging crashes"""
    try:
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"CareerRecommender_Log_{timestamp}.txt"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("=== Career Recommender Application Started ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Log file: {log_filename}")
        
        return logger
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return None

# Initialize logging
logger = setup_logging()

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    if logger:
        logger.info("Environment variables loaded successfully")
except ImportError:
    if logger:
        logger.info("python-dotenv not installed, skipping .env file loading")
    pass
except Exception as e:
    if logger:
        logger.error(f"Error loading environment variables: {e}")

# Import transformers with proper error handling and CPU fallback
try:
    import torch
    if logger:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        
    # Force CPU usage to avoid CUDA issues
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_default_tensor_type('torch.FloatTensor')
    
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    if logger:
        logger.info("Transformers library imported successfully - CPU mode forced")
        
except ImportError as e:
    if logger:
        logger.error(f"Failed to import transformers: {e}")
    transformers_available = False
except Exception as e:
    if logger:
        logger.error(f"Error with transformers setup: {e}")
    transformers_available = False
else:
    transformers_available = True

class CareerRecommender(QWidget):
    def __init__(self):
        super().__init__()
        
        try:
            if logger:
                logger.info("Initializing CareerRecommender widget")
            
            self.setWindowTitle("Career Path Recommender")
            self.setFixedSize(750, 650)
            self.setMinimumSize(750, 650)
            self.setMaximumSize(750, 650)

            # Initialize classifier as None
            self.classifier = None
            self.labels = []

            # Setup UI
            self.setup_ui()
            
            # Try to load model only if transformers is available
            if transformers_available:
                self.load_local_model()
            else:
                if logger:
                    logger.warning("Transformers not available - running in cloud-only mode")
                
        except Exception as e:
            if logger:
                logger.error(f"Error in __init__: {e}")
                logger.error(traceback.format_exc())
            self.show_error_message("Initialization Error", f"Failed to initialize application: {e}")

    def setup_ui(self):
        """Setup the user interface"""
        try:
            # Main layout
            main_layout = QVBoxLayout()

            self.label = QLabel("Enter your interests, hobbies, and strengths below:")
            main_layout.addWidget(self.label)

            self.text_input = QTextEdit()
            self.text_input.setMaximumHeight(120)
            main_layout.addWidget(self.text_input)

            # Add checkbox for cloud LLM
            self.use_cloud_checkbox = QCheckBox("Use Cloud LLM (Hugging Face) - More Accurate")
            self.use_cloud_checkbox.setChecked(True)
            main_layout.addWidget(self.use_cloud_checkbox)

            self.button = QPushButton("Get Career Recommendations")
            self.button.clicked.connect(self.safe_get_recommendations)
            main_layout.addWidget(self.button)

            # Create scrollable area for results
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            self.result_label = QLabel("Career Recommendations:")
            self.result_label.setWordWrap(True)
            self.result_label.setStyleSheet("padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f5f5f5; color: #333;")
            self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)

            scroll_area.setWidget(self.result_label)
            main_layout.addWidget(scroll_area)

            self.setLayout(main_layout)
            
            if logger:
                logger.info("UI setup completed successfully")
                
        except Exception as e:
            if logger:
                logger.error(f"Error setting up UI: {e}")
                logger.error(traceback.format_exc())
            raise

    def load_local_model(self):
        """Load the local BART model with comprehensive error handling"""
        try:
            if logger:
                logger.info("Attempting to load local BART model")
            
            # Create models directory
            models_dir = "./models"
            os.makedirs(models_dir, exist_ok=True)
            
            if logger:
                logger.info(f"Models directory: {os.path.abspath(models_dir)}")

            model_name = "facebook/bart-large-mnli"
            
            # Force CPU usage
            device = "cpu"
            if logger:
                logger.info(f"Using device: {device}")

            # Load with explicit CPU device mapping
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=models_dir,
                local_files_only=False
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                cache_dir=models_dir,
                local_files_only=False,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            self.classifier = pipeline(
                "zero-shot-classification", 
                model=model, 
                tokenizer=tokenizer,
                device=-1  # Force CPU
            )
            
            # Comprehensive career labels
            self.labels = [
                "Law & Legal Services",
                "Healthcare & Medicine", 
                "Education & Training",
                "Engineering & Technology",
                "Finance & Banking",
                "Management & Leadership",
                "Agriculture & Food Science",
                "Creative Arts & Design",
                "Scientific Research",
                "Media & Communication",
                "Public Administration",
                "Sales & Marketing",
                "Cybersecurity & IT",
                "Psychology & Counseling",
                "Hospitality & Tourism",
                "Logistics & Supply Chain",
                "Environmental Science",
                "Biotechnology & Life Sciences",
                "Journalism & Writing",
                "Aviation & Aerospace",
                "Sports & Fitness",
                "Social Work & Community Service",
                "Entrepreneurship & Business",
                "Architecture & Construction",
                "Automotive & Transportation",
                "Gaming & Entertainment",
                "Real Estate & Property",
                "Fashion & Textile",
                "Music & Audio Production",
                "Data Science & Analytics",
                "Human Resources",
                "Non-Profit & NGO Work",
                "International Relations",
                "Maritime & Naval",
                "Mining & Energy"
            ]
            
            if logger:
                logger.info("Local model loaded successfully")
                logger.info(f"Model device: {model.device if hasattr(model, 'device') else 'Unknown'}")
                
        except Exception as e:
            if logger:
                logger.error(f"Failed to load local model: {e}")
                logger.error(traceback.format_exc())
            
            self.classifier = None
            self.labels = []
            
            # Show warning but don't crash
            self.show_warning_message(
                "Model Loading Warning", 
                f"Could not load local AI model. The app will work with Cloud AI and Smart Analysis only.\n\nError: {str(e)}"
            )

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

    def get_cloud_recommendation(self, user_input):
        """Get recommendation from Hugging Face cloud LLM"""
        try:
            if logger:
                logger.info("Starting cloud recommendation process")
            
            # Check for token first
            token = self.get_hf_token()
            if not token:
                if logger:
                    logger.warning("No Hugging Face token found - cloud AI may not work")
            
            # Use a more reliable text generation model
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            prompt = f"""Career counselor analyzing user profile for career recommendations.

User Profile: "{user_input}"

Based on this profile, I recommend these careers:

1. """

            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
                if logger:
                    logger.info("Using authentication token for cloud API")

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            if logger:
                logger.info(f"Making API request to: {API_URL}")
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
            
            if logger:
                logger.info(f"API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    if generated_text and len(generated_text) > 20:
                        formatted_response = f"🌐 Cloud AI Recommendations:\n\n1. {generated_text}\n\n"
                        formatted_response += "💡 **For more accurate results, provide more detailed information about your interests, skills, and preferences.**"
                        if logger:
                            logger.info("Cloud recommendation generated successfully")
                        return formatted_response
            
            # Try alternative approach
            return self.try_alternative_cloud_api(user_input)
                
        except Exception as e:
            if logger:
                logger.error(f"Cloud API Error: {e}")
                logger.error(traceback.format_exc())
            return None

    def try_alternative_cloud_api(self, user_input):
        """Try alternative cloud API approaches"""
        try:
            if logger:
                logger.info("Trying alternative cloud API approach")
            
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            
            candidate_labels = [
                "Creative Arts Design", "Music Audio Production", "Engineering Technology",
                "Healthcare Medicine", "Education Training", "Finance Banking",
                "Sales Marketing", "Management Leadership", "Data Science Analytics"
            ]
            
            token = self.get_hf_token()
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            payload = {
                "inputs": user_input,
                "parameters": {
                    "candidate_labels": candidate_labels
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'labels' in result and 'scores' in result:
                    formatted_response = "🌐 Cloud AI Recommendations:\n\n"
                    
                    for i, (label, score) in enumerate(zip(result['labels'][:3], result['scores'][:3]), 1):
                        confidence = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
                        formatted_response += f"{i}. {label.replace('_', ' & ')}\n"
                        formatted_response += f"   Confidence: {confidence} ({score*100:.1f}%)\n"
                        formatted_response += f"   Reason: AI analysis shows strong alignment with this career domain\n\n"
                    
                    formatted_response += "💡 **For more accurate results, provide more detailed information about your interests, skills, and preferences.**"
                    if logger:
                        logger.info("Alternative cloud API succeeded")
                    return formatted_response
            
            return None
                
        except Exception as e:
            if logger:
                logger.error(f"Alternative Cloud API Error: {e}")
            return None

    def get_hf_token(self):
        """Get Hugging Face token from environment variable or file"""
        try:
            # Method 1: Environment variable
            token = os.environ.get('HUGGINGFACE_TOKEN')
            if token and token.strip():
                if logger:
                    logger.info("Found Hugging Face token in environment variable")
                return token.strip()
                
            # Method 2: From a token file
            token_file = os.path.join(os.path.dirname(__file__), 'hf_token.txt')
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                    if token:
                        if logger:
                            logger.info("Found Hugging Face token in file")
                        return token
            
            if logger:
                logger.info("No Hugging Face token found")
            return None
            
        except Exception as e:
            if logger:
                logger.error(f"Error reading Hugging Face token: {e}")
            return None

    def is_input_too_vague(self, user_input):
        """Check if user input is too vague to make meaningful recommendations"""
        user_lower = user_input.lower().strip()
        
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
            "• What specific activities do you enjoy doing in your free time?",
            "• What subjects or topics fascinate you the most?",
            "• What are your natural strengths or talents?",
            "• What type of work environment appeals to you?",
            "• Are there any fields you definitely want to avoid?",
            "• What motivates or excites you?",
            "• Do you prefer working with people, data, or objects?",
            "• What skills do you want to develop further?"
        ]
        
        result = "🤔 I need more specific information to provide accurate career recommendations.\n\n"
        result += "Your input appears to be quite general. To give you personalized recommendations, please provide more details about:\n\n"
        result += "\n".join(suggestions)
        result += "\n\n💡 Example of a good input:\n"
        result += '"I love solving mathematical problems, enjoy coding and building websites, '
        result += 'fascinated by artificial intelligence and data analysis. I have strong analytical '
        result += 'thinking skills and enjoy working independently. I\'m good with technology but '
        result += 'not great at public speaking or sales-oriented roles."\n\n'
        result += "Please try again with more specific details about your interests, strengths, and preferences!"
        
        return result

    def generate_smart_recommendation(self, user_input):
        """Generate intelligent recommendations based on user input analysis"""
        
        # Check if input is too vague first
        if self.is_input_too_vague(user_input):
            return self.generate_more_details_request(user_input)
        
        user_lower = user_input.lower()
        
        # Enhanced interest analysis with more comprehensive keywords
        interests = {
            'law': ['law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'justice', 'litigation', 'contract'],
            'healthcare': ['medicine', 'doctor', 'nurse', 'health', 'medical', 'hospital', 'patient', 'therapy', 'clinical'],
            'education': ['teach', 'education', 'school', 'student', 'learning', 'curriculum', 'academic', 'training'],
            'engineering': ['engineering', 'engineer', 'technical', 'mechanical', 'electrical', 'civil', 'software', 'systems'],
            'finance': ['finance', 'money', 'banking', 'investment', 'accounting', 'economics', 'financial', 'budget'],
            'management': ['management', 'leadership', 'team', 'project', 'organize', 'coordinate', 'supervise', 'strategy'],
            'agriculture': ['agriculture', 'farming', 'crops', 'livestock', 'food', 'rural', 'plant', 'harvest'],
            'creative': ['creative', 'art', 'design', 'artistic', 'imagination', 'visual', 'aesthetic', 'innovative'],
            'research': ['research', 'study', 'analyze', 'investigate', 'experiment', 'data', 'scientific', 'discovery'],
            'media': ['media', 'communication', 'broadcast', 'journalism', 'news', 'television', 'radio', 'digital'],
            'public_admin': ['government', 'public', 'policy', 'administration', 'civic', 'municipal', 'politics'],
            'sales': ['sales', 'marketing', 'customer', 'business', 'revenue', 'promotion', 'advertising', 'retail'],
            'cybersecurity': ['cybersecurity', 'security', 'hacking', 'network', 'information', 'cyber', 'protection'],
            'psychology': ['psychology', 'mental', 'behavior', 'counseling', 'therapy', 'mind', 'emotional', 'cognitive'],
            'hospitality': ['hospitality', 'hotel', 'tourism', 'travel', 'restaurant', 'service', 'guest', 'vacation'],
            'logistics': ['logistics', 'supply', 'chain', 'transportation', 'shipping', 'warehouse', 'distribution'],
            'environment': ['environment', 'environmental', 'ecology', 'conservation', 'sustainability', 'climate', 'green'],
            'biotech': ['biotechnology', 'biology', 'genetics', 'pharmaceutical', 'biomedical', 'life', 'molecular'],
            'journalism': ['journalism', 'writing', 'reporter', 'news', 'article', 'publish', 'editor', 'story'],
            'aviation': ['aviation', 'pilot', 'aircraft', 'flying', 'aerospace', 'airline', 'airport', 'flight'],
            'sports': ['sport', 'athletic', 'physical', 'exercise', 'fitness', 'competition', 'training', 'coaching'],
            'social_work': ['helping', 'community', 'social', 'volunteer', 'charity', 'people', 'service', 'support'],
            'music': ['music', 'song', 'instrument', 'audio', 'sound', 'musician', 'melody', 'rhythm'],
            'technology': ['technology', 'computer', 'coding', 'programming', 'tech', 'software', 'digital', 'IT'],
            'construction': ['construction', 'building', 'architecture', 'structural', 'blueprint', 'contractor'],
            'automotive': ['automotive', 'car', 'vehicle', 'mechanic', 'transportation', 'driving', 'engine'],
            'gaming': ['gaming', 'game', 'entertainment', 'video', 'interactive', 'developer', 'player'],
            'real_estate': ['real', 'estate', 'property', 'housing', 'land', 'broker', 'rental', 'investment'],
            'fashion': ['fashion', 'clothing', 'textile', 'style', 'apparel', 'designer', 'fabric', 'trend'],
            'data_science': ['data', 'analytics', 'statistics', 'machine', 'learning', 'algorithm', 'analysis'],
            'hr': ['human', 'resources', 'recruitment', 'hiring', 'employee', 'personnel', 'workplace'],
            'nonprofit': ['nonprofit', 'NGO', 'charity', 'foundation', 'cause', 'mission', 'humanitarian'],
            'international': ['international', 'global', 'foreign', 'diplomatic', 'relations', 'world', 'cultural']
        }
        
        # Career mapping to full names
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
        
        # Detailed career descriptions
        career_descriptions = {
            'Law & Legal Services': 'Legal practice, corporate law, criminal defense, civil rights, and judicial services',
            'Healthcare & Medicine': 'Medical practice, nursing, therapy, healthcare administration, and medical research',
            'Education & Training': 'Teaching, curriculum development, educational administration, and training programs',
            'Engineering & Technology': 'Software development, mechanical engineering, electrical systems, and technical innovation',
            'Finance & Banking': 'Financial analysis, banking services, investment management, and economic consulting',
            'Management & Leadership': 'Team leadership, project management, strategic planning, and organizational development',
            'Agriculture & Food Science': 'Farming, food production, agricultural research, and sustainable agriculture',
            'Creative Arts & Design': 'Graphic design, fine arts, creative direction, and artistic expression',
            'Scientific Research': 'Research and development, laboratory work, scientific analysis, and innovation',
            'Media & Communication': 'Broadcasting, digital media, public relations, and content creation',
            'Public Administration': 'Government services, policy development, public sector management, and civic engagement',
            'Sales & Marketing': 'Business development, digital marketing, customer relations, and brand management',
            'Cybersecurity & IT': 'Information security, network administration, system analysis, and cyber protection',
            'Psychology & Counseling': 'Mental health services, behavioral analysis, therapy, and psychological research',
            'Hospitality & Tourism': 'Hotel management, travel services, event planning, and customer service',
            'Logistics & Supply Chain': 'Supply chain management, transportation, warehouse operations, and distribution',
            'Environmental Science': 'Conservation, sustainability consulting, environmental research, and green technology',
            'Biotechnology & Life Sciences': 'Biomedical research, pharmaceutical development, and life sciences innovation',
            'Journalism & Writing': 'News reporting, content writing, editorial work, and media production',
            'Aviation & Aerospace': 'Pilot training, aerospace engineering, aviation management, and flight operations',
            'Sports & Fitness': 'Athletic training, sports management, fitness coaching, and recreational services',
            'Social Work & Community Service': 'Community outreach, social services, nonprofit work, and humanitarian aid',
            'Music & Audio Production': 'Music production, sound engineering, performance, and audio technology',
            'Data Science & Analytics': 'Data analysis, machine learning, statistical modeling, and business intelligence',
            'Architecture & Construction': 'Architectural design, construction management, urban planning, and building services',
            'Automotive & Transportation': 'Vehicle engineering, transportation systems, logistics, and automotive technology',
            'Gaming & Entertainment': 'Game development, entertainment production, interactive media, and digital content',
            'Real Estate & Property': 'Property management, real estate sales, investment analysis, and development',
            'Fashion & Textile': 'Fashion design, textile production, retail fashion, and style consulting',
            'Human Resources': 'Talent acquisition, employee relations, organizational development, and HR strategy',
            'Non-Profit & NGO Work': 'Charitable organizations, social impact, fundraising, and community development',
            'International Relations': 'Diplomacy, global affairs, international business, and cross-cultural communication'
        }
        
        # Detect negative mentions
        negative_phrases = ['not good at', 'bad at', 'terrible at', 'not that good', 'struggle with', 'hate', 'dislike']
        
        # Score interests
        interest_scores = {}
        total_matches = 0
        
        for category, keywords in interests.items():
            score = 0
            negative_score = 0
            
            for keyword in keywords:
                if keyword in user_lower:
                    # Check for negative context
                    keyword_pos = user_lower.find(keyword)
                    context = user_lower[max(0, keyword_pos-30):keyword_pos+30]
                    
                    if any(neg in context for neg in negative_phrases):
                        negative_score += 1
                    else:
                        score += 1
                        total_matches += 1
            
            # Reduce score if mentioned negatively
            final_score = max(0, score - negative_score * 2)
            if final_score > 0:
                interest_scores[category] = final_score
        
        # If very few matches found, ask for more details
        if total_matches < 3 or len(interest_scores) < 2:
            return self.generate_more_details_request(user_input)
        
        # Generate recommendations based on scores
        sorted_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for category, score in sorted_interests[:5]:  # Top 5 interests
            if category in career_mapping:
                career_name = career_mapping[category]
                confidence = 'High' if score >= 3 else 'Medium' if score >= 2 else 'Low'
                
                recommendations.append({
                    'title': career_name,
                    'confidence': confidence,
                    'score': score,
                    'reason': f"Strong alignment detected with {score} relevant keywords. {career_descriptions.get(career_name, 'Career matches your interests.')}"
                })
        
        # If still no good recommendations, ask for more details
        if not recommendations or all(rec['score'] <= 1 for rec in recommendations):
            return self.generate_more_details_request(user_input)
        
        # Format response
        result = "🧠 Smart Analysis Recommendations:\n\n"
        
        for i, rec in enumerate(recommendations[:3], 1):
            result += f"{i}. {rec['title']}\n"
            result += f"   Confidence: {rec['confidence']}\n"
            result += f"   Reason: {rec['reason']}\n\n"
        
        result += "💡 **For more accurate results, provide more detailed information about your interests, skills, and preferences.**"
        
        return result

    def get_local_recommendation(self, user_input):
        """Get recommendation from local model with comprehensive error handling"""
        try:
            if not self.classifier:
                if logger:
                    logger.warning("Local classifier not available")
                return "⚠️ Local AI model not available. Please use Cloud AI or Smart Analysis."
            
            if self.is_input_too_vague(user_input):
                return self.generate_more_details_request(user_input)
            
            if logger:
                logger.info("Running local model classification")
            
            result = self.classifier(user_input, self.labels)
            
            if result['scores'][0] < 0.3:
                return self.generate_more_details_request(user_input)
            
            # Improved filtering logic for negative context
            user_lower = user_input.lower()
            negative_phrases = ["not good at", "bad at", "terrible at", "struggle with", "not that good", "hate", "dislike"]
            
            # Enhanced career keywords mapping
            career_keywords = {
                "Sports & Fitness": ["sport", "sports", "athletic", "physical", "fitness", "exercise"],
                "Creative Arts & Design": ["art", "creative", "design", "artistic", "visual"],
                "Music & Audio Production": ["music", "audio", "sound", "musician", "song"],
                "Engineering & Technology": ["engineering", "technology", "technical", "software", "programming"],
                "Healthcare & Medicine": ["medicine", "health", "medical", "doctor", "nurse"],
                "Law & Legal Services": ["law", "legal", "lawyer", "attorney", "court"],
                "Finance & Banking": ["finance", "money", "banking", "financial", "investment"],
                "Education & Training": ["education", "teaching", "school", "training", "academic"],
                "Sales & Marketing": ["sales", "marketing", "business", "customer", "promotion"]
            }
            
            # Filter out careers mentioned with negative context
            filtered_recommendations = []
            for label, score in zip(result['labels'], result['scores']):
                skip_career = False
                
                if label in career_keywords:
                    for keyword in career_keywords[label]:
                        if keyword in user_lower:
                            # Check if mentioned with negative context
                            keyword_pos = user_lower.find(keyword)
                            context_before = user_lower[max(0, keyword_pos-50):keyword_pos]
                            context_after = user_lower[keyword_pos:keyword_pos+50]
                            full_context = context_before + context_after
                            
                            if any(neg_phrase in full_context for neg_phrase in negative_phrases):
                                skip_career = True
                                break
                
                if not skip_career:
                    filtered_recommendations.append((label, score))
            
            # Take top 3 from filtered results
            top_3 = filtered_recommendations[:3]
            
            # If no good recommendations after filtering
            if not top_3 or all(score < 0.25 for _, score in top_3):
                return self.generate_more_details_request(user_input)
            
            formatted_result = "🤖 AI Model Recommendations:\n\n"
            for i, (career, confidence) in enumerate(filtered_recommendations[:3], 1):
                formatted_result += f"{i}. {career}\n"
                formatted_result += f"   Confidence: {confidence*100:.1f}%\n"
                formatted_result += f"   Reason: AI classification based on comprehensive analysis of your input\n\n"
            
            formatted_result += "💡 **For more accurate results, provide more detailed information about your interests, skills, and preferences.**"
            
            if logger:
                logger.info("Local recommendation completed successfully")
            
            return formatted_result
            
        except Exception as e:
            if logger:
                logger.error(f"Error in local recommendation: {e}")
                logger.error(traceback.format_exc())
            return f"⚠️ Error with local AI model: {str(e)}\nPlease try Cloud AI or Smart Analysis."

    def get_recommendations(self):
        """Main recommendation method with enhanced error handling"""
        try:
            user_input = self.text_input.toPlainText().strip()

            if logger:
                logger.info(f"User input length: {len(user_input)}")

            if not user_input:
                self.show_warning_message("Input Needed", "Please enter some text.")
                return

            if len(user_input) < 5:
                self.show_warning_message("More Details Needed", 
                                          "Please provide more detailed information about your interests, "
                                          "hobbies, and strengths for better recommendations.")
                return

            self.result_label.setText("🔄 Analyzing your profile...\nPlease wait while we process your information...")
            QApplication.processEvents()

            if self.use_cloud_checkbox.isChecked():
                if logger:
                    logger.info("Using cloud AI analysis")
                
                self.result_label.setText("🌐 Connecting to cloud AI services...\nThis may take a moment...")
                QApplication.processEvents()
                
                cloud_result = self.get_cloud_recommendation(user_input)
                
                if cloud_result and len(cloud_result.strip()) > 10 and "Error" not in cloud_result:
                    self.result_label.setText(cloud_result)
                else:
                    if logger:
                        logger.info("Cloud AI failed, falling back to smart analysis")
                    
                    self.result_label.setText("⚠️ Cloud AI temporarily unavailable. Using Smart Analysis...\n")
                    QApplication.processEvents()
                    smart_result = self.generate_smart_recommendation(user_input)
                    self.result_label.setText(smart_result)
            else:
                if logger:
                    logger.info("Using local model analysis")
                
                local_result = self.get_local_recommendation(user_input)
                self.result_label.setText(local_result)
                
        except Exception as e:
            if logger:
                logger.error(f"Error in get_recommendations: {e}")
                logger.error(traceback.format_exc())
            raise  # Re-raise to be caught by safe_get_recommendations

# ...existing code for other methods...

if __name__ == "__main__":
    try:
        if logger:
            logger.info("Starting application")
        
        app = QApplication(sys.argv)
        window = CareerRecommender()
        window.show()
        
        if logger:
            logger.info("Application started successfully")
        
        sys.exit(app.exec())
        
    except Exception as e:
        if logger:
            logger.error(f"Critical application error: {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"Critical error: {e}")
            traceback.print_exc()
