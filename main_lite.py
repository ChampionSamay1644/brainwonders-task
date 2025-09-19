import sys
import os
import logging
import traceback
import re
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt

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

            subtitle = QLabel("Developed as part of BrainWonders Internship | Advanced Offline Analysis with 60+ Career Domains")
            subtitle.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-bottom: 15px;")
            layout.addWidget(subtitle)

            self.label = QLabel("Enter your interests, hobbies, strengths, and preferences below:")
            self.label.setStyleSheet("font-weight: bold; margin: 5px;")
            layout.addWidget(self.label)

            self.text_input = QTextEdit()
            self.text_input.setMaximumHeight(100)
            self.text_input.setPlaceholderText("Example: I love music and am very creative. I enjoy designing things and have a good eye for aesthetics. I'm interested in technology but I'm not that good at sports...")
            layout.addWidget(self.text_input)

            # Analysis method label
            method_label = QLabel("Analysis Method:")
            method_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            layout.addWidget(method_label)

            smart_label = QLabel("🧠 Advanced Offline Analysis Engine (Enhanced Pattern Recognition)")
            smart_label.setStyleSheet("font-size: 12px; color: #2c3e50; margin-left: 5px; margin-bottom: 10px;")
            layout.addWidget(smart_label)

            self.button = QPushButton("🔍 Get Career Recommendations")
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

    def generate_smart_recommendation(self, user_input):
        """Generate intelligent recommendations based on user input analysis"""
        
        # Check if input is too vague first
        if self.is_input_too_vague(user_input):
            return self.generate_more_details_request(user_input)

        user_lower = user_input.lower()
        
        # Enhanced text preprocessing
        def preprocess_text(text):
            """Enhanced text preprocessing with better normalization"""
            # Handle contractions
            contractions = {
                "don't": "do not", "can't": "cannot", "won't": "will not", "shouldn't": "should not",
                "wouldn't": "would not", "couldn't": "could not", "isn't": "is not", "aren't": "are not",
                "wasn't": "was not", "weren't": "were not", "haven't": "have not", "hasn't": "has not",
                "hadn't": "had not", "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
                "it's": "it is", "we're": "we are", "they're": "they are", "i've": "i have",
                "you've": "you have", "we've": "we have", "they've": "they have", "i'll": "i will",
                "you'll": "you will", "he'll": "he will", "she'll": "she will", "it'll": "it will",
                "we'll": "we will", "they'll": "they will", "i'd": "i would", "you'd": "you would",
                "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would"
            }
            
            # Expand contractions
            for contraction, expansion in contractions.items():
                text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
            
            # Normalize punctuation and spacing
            text = re.sub(r'[.,;!?]+', ' ', text)  # Replace punctuation with spaces
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip()
            
            return text
        
        # Apply preprocessing
        processed_text = preprocess_text(user_lower)
        
        if logger:
            logger.info(f"[PREPROCESSING] Original: '{user_lower[:100]}...'")
            logger.info(f"[PREPROCESSING] Processed: '{processed_text[:100]}...'")
        
        def detect_negation_scope(text, keyword_pos, keyword_len):
            """Detect if a keyword is within the scope of negation - Enhanced for complex expressions"""
            # Look for negation words before the keyword (extended range for complex phrases)
            pre_context_start = max(0, keyword_pos - 150)  # Increased range for longer negative expressions
            pre_context = text[pre_context_start:keyword_pos]
            
            # Enhanced negation indicators including complex phrases
            negation_words = [
                'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor',
                'barely', 'hardly', 'scarcely', 'seldom', 'rarely', 'without',
                # Added strong rejection phrases
                "wouldn't", "would never", "refuse to", "opposed to", "against",
                "can't stand", "hate", "dislike", "despise", "detest", "loathe"
            ]
            
            # Special patterns for very strong negative expressions
            strong_negative_patterns = [
                r'wouldn\'t\s+be\s+caught\s+.*?\s+even\s+if.*?life\s+depended',
                r'would\s+never\s+.*?\s+even\s+if',
                r'not\s+in\s+a\s+million\s+years',
                r'over\s+my\s+dead\s+body',
                r'absolutely\s+refuse',
                r'completely\s+against',
                r'wouldn\'t\s+be\s+caught\s+\w+',  # Catches "wouldn't be caught [word]"
                r'would\s+never\s+\w+'  # Catches "would never [word]"
            ]
            
            # Check for very strong negative patterns in extended context
            extended_context = text[max(0, keyword_pos - 200):keyword_pos + keyword_len + 50]
            for pattern in strong_negative_patterns:
                if re.search(pattern, extended_context, re.IGNORECASE):
                    if logger:
                        logger.debug(f"Strong negative pattern detected: {pattern}")
                    return True
            
            # Check for negation within the scope
            words_before = pre_context.split()[-15:]  # Increased to last 15 words before keyword
            
            for i, word in enumerate(words_before):
                # Clean word for better matching
                clean_word = re.sub(r'[^\w\']', '', word.lower())
                
                if any(neg_word in clean_word for neg_word in negation_words):
                    # Check if there's a clause boundary that would break negation scope
                    remaining_words = words_before[i+1:]
                    clause_breakers = ['but', 'however', 'although', 'though', 'while', 'whereas', 'except', 'and']
                    
                    # For very strong negatives, ignore clause breakers within short range
                    strong_negatives = ["wouldn't", "would never", "refuse", "hate", "despise"]
                    if any(strong_neg in clean_word for strong_neg in strong_negatives):
                        if len(remaining_words) <= 8:  # Within 8 words, negation still applies
                            return True
                    
                    # If no clause breaker found, keyword is in negation scope
                    if not any(breaker in ' '.join(remaining_words).lower() for breaker in clause_breakers):
                        return True
                        
            return False
        
        # Comprehensive keyword analysis - Expanded to 60+ career domains
        # Advanced keyword mapping with industry-specific terminology
        interests = {
            # Traditional Professional Fields
            'law': ['law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'justice', 'litigation', 'contract', 'rights', 'paralegal', 'advocacy'],
            'healthcare': ['medicine', 'doctor', 'nurse', 'health', 'medical', 'hospital', 'patient', 'therapy', 'clinical', 'wellness', 'pharmacy', 'dentistry'],
            'education': ['teach', 'education', 'school', 'student', 'learning', 'curriculum', 'academic', 'training', 'professor', 'tutor', 'instruction'],
            'engineering': ['engineering', 'engineer', 'technical', 'mechanical', 'electrical', 'civil', 'software', 'systems', 'development', 'design'],
            'finance': ['finance', 'money', 'banking', 'investment', 'accounting', 'economics', 'financial', 'budget', 'trading', 'audit', 'insurance'],
            'management': ['management', 'leadership', 'team', 'project', 'organize', 'coordinate', 'supervise', 'strategy', 'executive', 'director'],
            
            # Specialized Technical Fields
            'cybersecurity': ['cybersecurity', 'security', 'hacking', 'network', 'information', 'cyber', 'protection', 'privacy', 'firewall', 'encryption'],
            'data_science': ['data', 'analytics', 'statistics', 'machine', 'learning', 'algorithm', 'analysis', 'AI', 'artificial', 'intelligence', 'mining'],
            'software_development': ['programming', 'coding', 'developer', 'software', 'app', 'web', 'mobile', 'frontend', 'backend', 'fullstack'],
            'cloud_computing': ['cloud', 'AWS', 'azure', 'devops', 'infrastructure', 'deployment', 'kubernetes', 'docker', 'microservices'],
            'blockchain': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'smart', 'contracts', 'defi', 'web3', 'crypto'],
            'robotics': ['robotics', 'automation', 'robot', 'mechanical', 'sensors', 'actuators', 'autonomous', 'manufacturing'],
            'aerospace': ['aerospace', 'aviation', 'pilot', 'aircraft', 'flying', 'airline', 'airport', 'flight', 'navigation', 'satellite'],
            'biotechnology': ['biotechnology', 'biology', 'genetics', 'pharmaceutical', 'biomedical', 'life', 'molecular', 'biochemistry', 'bioengineering'],
            
            # Creative and Design Fields
            'graphic_design': ['graphic', 'design', 'visual', 'creative', 'logo', 'branding', 'typography', 'illustration', 'photoshop', 'art'],
            'web_design': ['web', 'design', 'UI', 'UX', 'user', 'interface', 'experience', 'wireframe', 'prototype', 'figma'],
            'animation': ['animation', 'animator', '3D', '2D', 'cartoon', 'motion', 'graphics', 'maya', 'blender', 'after', 'effects', 'art'],
            'photography': ['photography', 'photographer', 'camera', 'photo', 'portrait', 'wedding', 'commercial', 'editing', 'lightroom', 'art'],
            'videography': ['videography', 'video', 'filming', 'cinematography', 'editing', 'premiere', 'final', 'cut', 'production'],
            'fashion_design': ['fashion', 'design', 'clothing', 'textile', 'style', 'apparel', 'designer', 'fabric', 'trend', 'couture', 'art'],
            'interior_design': ['interior', 'design', 'decoration', 'furniture', 'space', 'residential', 'commercial', 'aesthetics'],
            'architecture': ['architecture', 'architect', 'building', 'construction', 'structural', 'blueprint', 'design', 'planning'],
            'music_production': ['music', 'production', 'audio', 'sound', 'recording', 'mixing', 'mastering', 'studio', 'composer'],
            'game_design': ['game', 'design', 'gaming', 'unity', 'unreal', 'level', 'character', 'gameplay', 'mechanics'],
            
            # Media and Communication
            'journalism': ['journalism', 'writing', 'reporter', 'news', 'article', 'publish', 'editor', 'story', 'interview', 'media'],
            'content_creation': ['content', 'creator', 'youtube', 'social', 'media', 'influencer', 'blogging', 'podcast', 'streaming'],
            'digital_marketing': ['marketing', 'digital', 'SEO', 'social', 'media', 'advertising', 'campaign', 'brand', 'analytics'],
            'public_relations': ['public', 'relations', 'PR', 'communication', 'reputation', 'crisis', 'management', 'press'],
            'broadcasting': ['broadcasting', 'radio', 'television', 'TV', 'anchor', 'host', 'producer', 'live', 'streaming'],
            'translation': ['translation', 'interpreter', 'language', 'multilingual', 'localization', 'linguistic', 'foreign'],
            
            # Business and Sales
            'sales': ['sales', 'selling', 'customer', 'business', 'revenue', 'promotion', 'retail', 'commerce', 'negotiation'],
            'consulting': ['consulting', 'consultant', 'advisory', 'strategy', 'business', 'solutions', 'analysis', 'recommendations'],
            'entrepreneurship': ['entrepreneur', 'startup', 'business', 'owner', 'venture', 'innovation', 'risk', 'investment'],
            'real_estate': ['real', 'estate', 'property', 'housing', 'land', 'broker', 'rental', 'investment', 'commercial'],
            'human_resources': ['human', 'resources', 'HR', 'recruitment', 'hiring', 'employee', 'personnel', 'workplace', 'talent'],
            'supply_chain': ['supply', 'chain', 'logistics', 'procurement', 'warehouse', 'distribution', 'inventory', 'shipping'],
            
            # Research and Science
            'research': ['research', 'study', 'analyze', 'investigate', 'experiment', 'data', 'scientific', 'discovery', 'analysis'],
            'psychology': ['psychology', 'mental', 'behavior', 'counseling', 'therapy', 'mind', 'emotional', 'cognitive', 'behavioral'],
            'environmental_science': ['environment', 'environmental', 'ecology', 'conservation', 'sustainability', 'climate', 'green', 'renewable'],
            'chemistry': ['chemistry', 'chemical', 'laboratory', 'compound', 'reaction', 'molecular', 'analytical', 'organic'],
            'physics': ['physics', 'quantum', 'particle', 'energy', 'mechanics', 'theoretical', 'experimental', 'nuclear'],
            'mathematics': ['mathematics', 'math', 'calculus', 'algebra', 'geometry', 'statistics', 'probability', 'equations'],
            'astronomy': ['astronomy', 'space', 'cosmos', 'stars', 'planets', 'telescope', 'astrophysics', 'universe'],
            'geology': ['geology', 'earth', 'rocks', 'minerals', 'geological', 'mining', 'petroleum', 'seismic'],
            
            # Service Industries
            'hospitality': ['hospitality', 'hotel', 'tourism', 'travel', 'restaurant', 'service', 'guest', 'vacation', 'event'],
            'culinary': ['culinary', 'cooking', 'chef', 'kitchen', 'restaurant', 'food', 'cuisine', 'baking', 'pastry'],
            'fitness': ['fitness', 'personal', 'trainer', 'gym', 'exercise', 'health', 'nutrition', 'wellness', 'sports', 'athletics', 'athletic'],
            'beauty': ['beauty', 'cosmetics', 'makeup', 'skincare', 'salon', 'aesthetics', 'hair', 'styling'],
            'childcare': ['childcare', 'children', 'daycare', 'nanny', 'early', 'childhood', 'development', 'kids'],
            
            # Government and Public Service
            'public_administration': ['government', 'public', 'policy', 'administration', 'civic', 'municipal', 'politics', 'service'],
            'law_enforcement': ['police', 'law', 'enforcement', 'security', 'detective', 'investigation', 'criminal', 'justice'],
            'firefighting': ['firefighter', 'fire', 'emergency', 'rescue', 'paramedic', 'first', 'responder', 'safety'],
            'social_work': ['social', 'work', 'helping', 'community', 'volunteer', 'charity', 'people', 'service', 'support', 'welfare'],
            
            # Agriculture and Nature
            'agriculture': ['agriculture', 'farming', 'crops', 'livestock', 'food', 'rural', 'plant', 'harvest', 'organic'],
            'forestry': ['forestry', 'forest', 'trees', 'conservation', 'wildlife', 'ranger', 'natural', 'resources'],
            'veterinary': ['veterinary', 'animal', 'pet', 'wildlife', 'zoo', 'veterinarian', 'animal', 'care'],
            
            # Transportation and Logistics
            'automotive': ['automotive', 'car', 'vehicle', 'mechanic', 'transportation', 'driving', 'engine', 'repair'],
            'maritime': ['maritime', 'shipping', 'ocean', 'naval', 'port', 'cargo', 'marine', 'logistics'],
            'railway': ['railway', 'train', 'transportation', 'locomotive', 'rail', 'transit', 'conductor'],
            
            # Emerging Fields
            'sustainability': ['sustainability', 'renewable', 'energy', 'solar', 'wind', 'green', 'environmental', 'carbon'],
            'virtual_reality': ['virtual', 'reality', 'VR', 'AR', 'augmented', 'immersive', 'simulation', 'metaverse'],
            'nanotechnology': ['nanotechnology', 'nano', 'microscopic', 'molecular', 'materials', 'science'],
            'space_technology': ['space', 'technology', 'satellite', 'rocket', 'mars', 'exploration', 'astronaut'],
            
            # Arts and Entertainment
            'performing_arts': ['performing', 'arts', 'theater', 'acting', 'drama', 'stage', 'performance', 'actor', 'art'],
            'dance': ['dance', 'ballet', 'choreography', 'movement', 'rhythm', 'performance', 'instructor'],
            'literature': ['literature', 'writing', 'author', 'novel', 'poetry', 'creative', 'writing', 'publishing'],
            'film_production': ['film', 'movie', 'cinema', 'director', 'producer', 'screenplay', 'editing', 'camera']
        }
        
        # Comprehensive career mapping - 60+ Professional domains
        career_mapping = {
            # Traditional Professional Fields
            'law': 'Law & Legal Services',
            'healthcare': 'Healthcare & Medicine',
            'education': 'Education & Training',
            'engineering': 'Engineering & Technology',
            'finance': 'Finance & Banking',
            'management': 'Management & Leadership',
            
            # Specialized Technical Fields
            'cybersecurity': 'Cybersecurity & Information Security',
            'data_science': 'Data Science & Analytics',
            'software_development': 'Software Development & Programming',
            'cloud_computing': 'Cloud Computing & DevOps',
            'blockchain': 'Blockchain & Cryptocurrency',
            'robotics': 'Robotics & Automation',
            'aerospace': 'Aerospace & Aviation',
            'biotechnology': 'Biotechnology & Life Sciences',
            
            # Creative and Design Fields
            'graphic_design': 'Graphic Design & Visual Arts',
            'web_design': 'Web Design & User Experience',
            'animation': 'Animation & Motion Graphics',
            'photography': 'Photography & Visual Media',
            'videography': 'Videography & Film Production',
            'fashion_design': 'Fashion Design & Textile Arts',
            'interior_design': 'Interior Design & Space Planning',
            'architecture': 'Architecture & Building Design',
            'music_production': 'Music Production & Audio Engineering',
            'game_design': 'Game Design & Development',
            
            # Media and Communication
            'journalism': 'Journalism & News Media',
            'content_creation': 'Content Creation & Digital Media',
            'digital_marketing': 'Digital Marketing & SEO',
            'public_relations': 'Public Relations & Communications',
            'broadcasting': 'Broadcasting & Media Production',
            'translation': 'Translation & Linguistics',
            
            # Business and Sales
            'sales': 'Sales & Business Development',
            'consulting': 'Business Consulting & Strategy',
            'entrepreneurship': 'Entrepreneurship & Innovation',
            'real_estate': 'Real Estate & Property Management',
            'human_resources': 'Human Resources & Talent Management',
            'supply_chain': 'Supply Chain & Logistics Management',
            
            # Research and Science
            'research': 'Scientific Research & Development',
            'psychology': 'Psychology & Mental Health',
            'environmental_science': 'Environmental Science & Sustainability',
            'chemistry': 'Chemistry & Chemical Sciences',
            'physics': 'Physics & Physical Sciences',
            'mathematics': 'Mathematics & Applied Math',
            'astronomy': 'Astronomy & Space Sciences',
            'geology': 'Geology & Earth Sciences',
            
            # Service Industries
            'hospitality': 'Hospitality & Tourism Management',
            'culinary': 'Culinary Arts & Food Service',
            'fitness': 'Fitness & Personal Training',
            'beauty': 'Beauty & Cosmetic Services',
            'childcare': 'Childcare & Early Development',
            
            # Government and Public Service
            'public_administration': 'Public Administration & Government',
            'law_enforcement': 'Law Enforcement & Criminal Justice',
            'firefighting': 'Emergency Services & Public Safety',
            'social_work': 'Social Work & Community Services',
            
            # Agriculture and Nature
            'agriculture': 'Agriculture & Food Production',
            'forestry': 'Forestry & Natural Resource Management',
            'veterinary': 'Veterinary Medicine & Animal Care',
            
            # Transportation and Logistics
            'automotive': 'Automotive & Transportation',
            'maritime': 'Maritime & Shipping Industry',
            'railway': 'Railway & Transit Systems',
            
            # Emerging Fields
            'sustainability': 'Sustainability & Renewable Energy',
            'virtual_reality': 'Virtual Reality & Immersive Technology',
            'nanotechnology': 'Nanotechnology & Materials Science',
            'space_technology': 'Space Technology & Exploration',
            
            # Arts and Entertainment
            'performing_arts': 'Performing Arts & Theater',
            'dance': 'Dance & Movement Arts',
            'literature': 'Literature & Creative Writing',
            'film_production': 'Film Production & Cinema'
        }
        
        # Advanced sentiment analysis - Enhanced for sarcasm, irony, and negation
        negative_patterns = [
            # Direct negation patterns
            r'\bnot\s+good\s+at\b', r'\bbad\s+at\b', r'\bterrible\s+at\b', r'\bnot\s+that\s+good\b',
            r'\bstruggle\s+with\b', r'\bhate\b', r'\bdislike\b', r'\bavoid\b', r'\bweak\s+in\b',
            r'\bpoor\s+at\b', r'\bdifficulty\s+with\b', r"\bcan't\b", r"\bdon't\b", r'\bunable\b',
            
            # Very strong rejection patterns (highest priority)
            r'\bwouldn\'t\s+be\s+caught\s+.*\s+even\s+if\s+my\s+life\s+depended\s+on\s+it\b',
            r'\bwould\s+never\s+.*\s+even\s+if\b', r'\bwouldn\'t\s+.*\s+if\s+you\s+paid\s+me\b',
            r'\bover\s+my\s+dead\s+body\b', r'\bnot\s+in\s+a\s+million\s+years\b',
            r'\bwild\s+horses\s+couldn\'t\s+drag\s+me\b', r'\babsolutely\s+refuse\b',
            
            # Strong negative expressions
            r'\bwouldn\'t\s+be\s+caught\b', r'\bwould\s+never\b', r'\bcompletely\s+against\b',
            r'\btotally\s+opposed\s+to\b', r'\bstrongly\s+dislike\b', r'\bcan\'t\s+stand\b',
            r'\bdetest\b', r'\babhor\b', r'\bloathe\b', r'\bdespise\b',
            
            # Sarcasm indicators
            r'\byeah\s+right\b', r'\bsure\b.*\bnot\b', r'\bobviously\s+not\b', r'\bof\s+course\s+not\b',
            r'\bgreat\b.*\bnot\b', r'\blove\b.*\bnot\b', r'\bperfect\b.*\bnot\b',
            
            # Irony and indirect negation
            r'\bif\s+only\b', r'\bwish\s+i\s+could\b', r'\bin\s+my\s+dreams\b', r'\bnever\s+gonna\b',
            r'\bimpossible\s+for\s+me\b', r'\bnot\s+my\s+thing\b', r'\bnot\s+for\s+me\b',
            
            # Intensity modifiers with negative context
            r'\babsolutely\s+not\b', r'\bdefinitely\s+not\b', r'\btotally\s+not\b',
            r'\bcompletely\s+hopeless\b', r'\butterly\s+useless\b'
        ]
        
        positive_patterns = [
            # Strong positive expressions
            r'\blove\b', r'\benjoy\b', r'\bpassionate\s+about\b', r'\bgood\s+at\b', r'\bexcel\s+at\b',
            r'\btalented\s+at\b', r'\bskilled\s+in\b', r'\binterested\s+in\b', r'\bfascinated\s+by\b',
            r'\bexcited\s+about\b', r'\bstrong\s+in\b', r'\bproficient\s+in\b', r'\bgreat\s+at\b',
            
            # Enthusiasm markers
            r'\badore\b', r'\babsolutely\s+love\b', r'\breally\s+enjoy\b', r'\btotally\s+into\b',
            r'\bcrazy\s+about\b', r'\bobsessed\s+with\b', r'\bcan\'t\s+get\s+enough\b',
            
            # Achievement indicators
            r'\baccomplished\b', r'\bsuccessful\b', r'\bproud\s+of\b', r'\bmastered\b',
            r'\bexpertise\b', r'\bspecialize\b', r'\bnatural\s+at\b'
        ]
        
        # Contextual modifiers
        doubt_indicators = [
            r'\bi\s+think\b', r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b', r'\bkind\s+of\b',
            r'\bsort\s+of\b', r'\bi\s+guess\b', r'\bprobably\b', r'\bsomewhat\b'
        ]
        
        confidence_boosters = [
            r'\babsolutely\b', r'\bdefinitely\b', r'\bcertainly\b', r'\btotally\b',
            r'\bcompletely\b', r'\breally\b', r'\btruly\b', r'\bextremely\b'
        ]

        def analyze_sentiment(text, keyword_pos, keyword_len):
            """Enhanced sentiment analysis with advanced negation detection"""
            # Get extended context around the keyword
            context_start = max(0, keyword_pos - 100)  # Increased context window
            context_end = min(len(text), keyword_pos + keyword_len + 100)
            context = text[context_start:context_end]
            
            # Get even broader context for very strong negative patterns  
            extended_context = text[max(0, keyword_pos - 300):keyword_pos + keyword_len + 200]
            
            sentiment_score = 1.0  # Neutral baseline
            
            # Find if there's a clause breaker between strong negatives and keyword
            keyword_text = text[keyword_pos:keyword_pos + keyword_len]
            
            # Find if there's a clause breaker between start of extended context and keyword
            pre_keyword_text = text[max(0, keyword_pos - 300):keyword_pos]
            clause_breakers = ['but', 'however', 'although', 'though', 'while', 'whereas', 'except']
            
            # Find the last clause breaker before the keyword in the full text
            last_clause_breaker_absolute_pos = -1
            for breaker in clause_breakers:
                # Search in the range from 300 chars before keyword to keyword position
                search_start = max(0, keyword_pos - 300)
                search_text = text[search_start:keyword_pos]
                breaker_matches = list(re.finditer(rf'\b{breaker}\b', search_text, re.IGNORECASE))
                if breaker_matches:
                    # Convert relative position to absolute position in the text
                    absolute_pos = search_start + breaker_matches[-1].end()
                    last_clause_breaker_absolute_pos = max(last_clause_breaker_absolute_pos, absolute_pos)
            
            # If there's a clause breaker, only consider context after it for strong negatives
            if last_clause_breaker_absolute_pos > -1:
                # Start context from after the clause breaker
                clause_adjusted_context = text[last_clause_breaker_absolute_pos:keyword_pos + keyword_len + 200]
            else:
                clause_adjusted_context = extended_context
            
            # Check for negation scope first
            if detect_negation_scope(text, keyword_pos, keyword_len):
                sentiment_score -= 2.5  # Increased negation penalty
                if logger:
                    logger.debug(f"Negation scope detected for keyword at position {keyword_pos}")
            
            # Special check for "wouldn't be caught [keyword]" pattern (with and without apostrophe)
            # Use clause-adjusted context for this check
            wouldnt_patterns = [
                rf'wouldn\'t\s+be\s+caught\s+.*?\b{re.escape(keyword_text)}\b',
                rf'wouldnt\s+be\s+caught\s+.*?\b{re.escape(keyword_text)}\b',
                rf'wouldn\'t\s+be\s+caught\s+{re.escape(keyword_text)}\b',
                rf'wouldnt\s+be\s+caught\s+{re.escape(keyword_text)}\b'
            ]
            
            for pattern in wouldnt_patterns:
                if re.search(pattern, clause_adjusted_context, re.IGNORECASE):
                    sentiment_score -= 10.0  # Massive penalty for this specific pattern
                    if logger:
                        logger.debug(f"'Wouldn't be caught [keyword]' pattern detected for: {keyword_text} using pattern: {pattern}")
                    break
            
            # Check for very strong negative patterns first (highest priority)
            # Use clause-adjusted context for this check
            very_strong_negatives = [
                rf'wouldn\'t\s+be\s+caught\s+.*?\b{re.escape(keyword_text)}\b.*?even\s+if.*?life\s+depended',
                rf'wouldnt\s+be\s+caught\s+.*?\b{re.escape(keyword_text)}\b.*?even\s+if.*?life\s+depended',
                rf'would\s+never\s+.*?\b{re.escape(keyword_text)}\b.*?\s+even\s+if',
                rf'wouldn\'t\s+.*?\b{re.escape(keyword_text)}\b.*?\s+if\s+you\s+paid\s+me',
                rf'wouldnt\s+.*?\b{re.escape(keyword_text)}\b.*?\s+if\s+you\s+paid\s+me',
                rf'over\s+my\s+dead\s+body.*?\b{re.escape(keyword_text)}\b',
                rf'not\s+in\s+a\s+million\s+years.*?\b{re.escape(keyword_text)}\b',
                rf'absolutely\s+refuse.*?\b{re.escape(keyword_text)}\b',
                rf'completely\s+against.*?\b{re.escape(keyword_text)}\b',
                rf'wouldn\'t\s+be\s+caught\s+\b{re.escape(keyword_text)}\b',  # Direct pattern
                rf'wouldnt\s+be\s+caught\s+\b{re.escape(keyword_text)}\b',  # Direct pattern without apostrophe
            ]
            
            for pattern in very_strong_negatives:
                if re.search(pattern, clause_adjusted_context, re.IGNORECASE):
                    sentiment_score -= 10.0  # Massive penalty for very strong rejection
                    if logger:
                        logger.debug(f"Very strong negative pattern found: {pattern} for keyword: {keyword_text}")
            
            # Check for strong negative patterns
            strong_negatives = [
                rf'would\s+never\s+\b{re.escape(keyword_text)}\b',  # "would never [keyword]" 
                rf'hate\s+\b{re.escape(keyword_text)}\b', rf'despise\s+\b{re.escape(keyword_text)}\b',
                rf'can\'t\s+stand\s+\b{re.escape(keyword_text)}\b', rf'detest\s+\b{re.escape(keyword_text)}\b'
            ]
            
            for pattern in strong_negatives:
                if re.search(pattern, clause_adjusted_context, re.IGNORECASE):
                    sentiment_score -= 8.0  # Very strong negative penalty
                    if logger:
                        logger.debug(f"Strong negative pattern found: {pattern} for keyword: {keyword_text}")
            
            # Check for regular negative patterns using regex
            for pattern in negative_patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    sentiment_score -= 1.5  # Increased from 1.2
                    if logger:
                        logger.debug(f"Negative pattern found: {pattern} in context: {context}")
            
            # Check for positive patterns
            for pattern in positive_patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    sentiment_score += 0.9
                    if logger:
                        logger.debug(f"Positive pattern found: {pattern} in context: {context}")
            
            # Apply doubt modifiers
            for pattern in doubt_indicators:
                if re.search(pattern, context, re.IGNORECASE):
                    sentiment_score *= 0.6
                    if logger:
                        logger.debug(f"Doubt indicator found: {pattern}")
            
            # Apply confidence boosters
            for pattern in confidence_boosters:
                if re.search(pattern, context, re.IGNORECASE):
                    sentiment_score *= 1.4
                    if logger:
                        logger.debug(f"Confidence booster found: {pattern}")
            
            # Enhanced sarcasm detection
            sarcasm_patterns = [
                r'\byeah\s+right\b', r'\bsure\s+thing\b', r'\bof\s+course\b.*\bnot\b',
                r'\bobviously\b.*\bnot\b', r'\btotally\b.*\bnot\b'
            ]
            
            for pattern in sarcasm_patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    sentiment_score -= 2.5  # Increased sarcasm penalty
                    if logger:
                        logger.debug(f"Sarcasm detected: {pattern}")
            
            # Ensure score can go negative for strong rejections
            return max(-15.0, sentiment_score)  # Allow very strong negative scores
        
        # Intelligent scoring algorithm with enhanced text processing
        interest_scores = {}
        total_matches = 0
        
        for category, keywords in interests.items():
            score = 0
            detailed_matches = []
            
            for keyword in keywords:
                # Search for whole word matches using regex with word boundaries
                original_match = re.search(rf'\b{re.escape(keyword)}\b', user_lower)
                processed_match = re.search(rf'\b{re.escape(keyword)}\b', processed_text)
                
                if original_match or processed_match:
                    # Use the position from original text for context analysis
                    keyword_pos = original_match.start() if original_match else processed_match.start()
                    
                    # Apply enhanced sentiment analysis
                    sentiment_score = analyze_sentiment(user_lower, keyword_pos, len(keyword))
                    
                    # Enhanced keyword scoring with frequency consideration
                    frequency_bonus = 1.0
                    
                    # Count multiple occurrences using word boundaries
                    keyword_matches = re.findall(rf'\b{re.escape(keyword)}\b', user_lower)
                    keyword_count = len(keyword_matches)
                    if keyword_count > 1:
                        frequency_bonus = 1.0 + (keyword_count - 1) * 0.3  # Diminishing returns
                    
                    # Calculate final keyword score
                    keyword_score = sentiment_score * frequency_bonus
                    score += keyword_score
                    total_matches += 1
                    
                    detailed_matches.append({
                        'keyword': keyword,
                        'sentiment': sentiment_score,
                        'frequency': keyword_count,
                        'score': keyword_score
                    })
                    
                    if logger:
                        logger.debug(f"Category: {category}, Keyword: '{keyword}', "
                                   f"Sentiment: {sentiment_score:.2f}, Frequency: {keyword_count}, "
                                   f"Final Score: {keyword_score:.2f}")
            
            # Store results if we have matches
            if score > 0:
                interest_scores[category] = {
                    'total_score': score,
                    'matches': detailed_matches,
                    'avg_sentiment': score / len(detailed_matches) if detailed_matches else 0,
                    'diversity_score': len(detailed_matches)  # Number of different keywords matched
                }
            
            if logger:
                logger.debug(f"[{category}] total_score={score:.2f}, matches={len(detailed_matches)}")
        
        # Enhanced filtering with diversity consideration
        if total_matches < 3 or len(interest_scores) < 2:
            return self.generate_more_details_request(user_input)
        
        # Filter out categories with negative sentiment (indicating rejection)
        filtered_scores = {}
        for category, data in interest_scores.items():
            # Only include categories with positive average sentiment
            if data['avg_sentiment'] > 0:
                filtered_scores[category] = data
            elif logger:
                logger.info(f"Filtering out {category} due to negative sentiment: {data['avg_sentiment']:.2f}")
        
        # Generate top recommendations with enhanced analysis
        sorted_interests = sorted(filtered_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        recommendations = []
        for category, data in sorted_interests[:6]:  # Top 6 for better diversity
            if category in career_mapping:
                career_name = career_mapping[category]
                total_score = data['total_score']
                avg_sentiment = data['avg_sentiment']
                match_count = len(data['matches'])
                
                # Enhanced confidence scoring
                if total_score >= 5.0 and avg_sentiment >= 1.2:
                    confidence = 'Very High'
                elif total_score >= 3.0 and avg_sentiment >= 1.0:
                    confidence = 'High'
                elif total_score >= 2.0 and avg_sentiment >= 0.8:
                    confidence = 'Medium'
                elif total_score >= 1.0:
                    confidence = 'Moderate'
                else:
                    confidence = 'Low'
                
                # Generate detailed reasoning
                top_keywords = [match['keyword'] for match in sorted(data['matches'], 
                                                                   key=lambda x: x['score'], reverse=True)[:3]]
                
                sentiment_desc = "strong positive indicators" if avg_sentiment >= 1.2 else \
                               "positive alignment" if avg_sentiment >= 1.0 else \
                               "moderate interest signals" if avg_sentiment >= 0.8 else \
                               "basic interest indicators"
                
                reason = (f"Detected {match_count} relevant career indicators with {sentiment_desc}. "
                         f"Top matches: {', '.join(top_keywords)}. Advanced sentiment analysis "
                         f"reveals genuine interest pattern (sentiment score: {avg_sentiment:.2f}).")
                
                recommendations.append({
                    'title': career_name,
                    'confidence': confidence,
                    'score': total_score,
                    'sentiment': avg_sentiment,
                    'reason': reason,
                    'match_count': match_count
                })
        
        # Enhanced filtering for quality recommendations - exclude negative sentiments
        quality_recommendations = [rec for rec in recommendations 
                                 if rec['score'] >= 1.0 and rec['sentiment'] >= 0.5]  # Higher sentiment threshold
        
        if not quality_recommendations:
            return self.generate_more_details_request(user_input)
        
        # Professional formatting with enhanced details
        result = "🧠 Advanced Offline Analysis Results:\n\n"
        result += "✨ Using sophisticated sentiment analysis, pattern recognition, and linguistic processing:\n\n"
        
        for i, rec in enumerate(quality_recommendations[:4], 1):  # Top 4 quality matches
            confidence_emoji = "🎯" if rec['confidence'] == 'Very High' else \
                             "⭐" if rec['confidence'] == 'High' else \
                             "✅" if rec['confidence'] == 'Medium' else "💡"
            
            result += f"{confidence_emoji} {i}. {rec['title']}\n"
            result += f"   Confidence Level: {rec['confidence']} (Score: {rec['score']:.1f}, Sentiment: {rec['sentiment']:.2f})\n"
            result += f"   Analysis: {rec['reason']}\n"
            result += f"   Pattern Strength: {rec['match_count']} keyword matches with contextual sentiment validation\n\n"
        
        result += "� **Analysis Features:**\n"
        result += "• Advanced sentiment analysis detecting sarcasm, irony, and negation\n"
        result += "• 60+ career domain coverage with specialized keyword recognition\n"
        result += "• Contextual understanding of positive/negative expressions\n"
        result += "• Multi-layered confidence scoring based on linguistic patterns\n\n"
        result += "💡 **Results are based on sophisticated offline NLP processing - no internet required!**"
        
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

            self.result_label.setText("🔄 Analyzing your profile with advanced pattern recognition...\n\n⏳ Processing linguistic patterns and sentiment...")
            QApplication.processEvents()

            if logger:
                logger.info("Using offline smart analysis")
            
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
