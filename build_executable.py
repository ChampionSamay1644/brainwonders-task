"""
Executable Build Script for AI Career Recommender
BrainWonders Internship Project

This script creates a standalone executable without the heavy BART model,
making it suitable for distribution while maintaining core AI functionality.
"""

import os
import shutil
import subprocess
import sys

# ANSI color codes for better visibility
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

def print_colored(text, color=Colors.WHITE, end='\n'):
    """Print colored text with fallback for systems without color support"""
    try:
        print(f"{color}{text}{Colors.RESET}", end=end)
    except:
        print(text, end=end)

def print_header(text):
    """Print a header with emphasis"""
    print_colored("=" * 60, Colors.CYAN)
    print_colored(text, Colors.BOLD + Colors.CYAN)
    print_colored("=" * 60, Colors.CYAN)

def print_success(text):
    """Print success message"""
    print_colored(f"✅ {text}", Colors.GREEN)

def print_error(text):
    """Print error message"""
    print_colored(f"❌ {text}", Colors.RED)

def print_warning(text):
    """Print warning message"""
    print_colored(f"⚠️ {text}", Colors.YELLOW)

def print_info(text):
    """Print info message"""
    print_colored(f"ℹ️ {text}", Colors.BLUE)

def print_step(text):
    """Print step message"""
    print_colored(f"🔨 {text}", Colors.MAGENTA)

def install_pyinstaller():
    """Install PyInstaller with proper error handling"""
    try:
        print_step("Installing PyInstaller...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                              capture_output=True, text=True, check=True)
        print_success("PyInstaller installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install PyInstaller: {e}")
        if e.stderr:
            print_colored(f"Error details: {e.stderr}", Colors.RED)
        return False

def check_pyinstaller():
    """Check if PyInstaller is available and working"""
    try:
        # Test if pyinstaller command works
        result = subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success(f"PyInstaller found: {result.stdout.strip()}")
            return True
    except Exception as e:
        print_warning(f"PyInstaller check failed: {e}")
    
    try:
        # Alternative: check if we can import PyInstaller
        import PyInstaller
        print_success("PyInstaller module found")
        return True
    except ImportError as e:
        print_warning(f"PyInstaller import failed: {e}")
    
    return False

def create_executable():
    """Create lightweight executable for distribution"""
    
    print_header("AI Career Recommender - Build System")
    print_colored("🎓 BrainWonders Internship Project", Colors.BOLD)
    print()
    
    # Check if main_lite.py exists
    if not os.path.exists("main_lite.py"):
        print_error("main_lite.py not found. Please create it first.")
        print_info("The lightweight version should be created to exclude heavy BART model.")
        return False
    
    print_success("Found main_lite.py")
    
    # Check PyInstaller availability
    if not check_pyinstaller():
        print_error("PyInstaller not found. Attempting to install...")
        if not install_pyinstaller():
            print()
            print_colored("💡 Manual Installation Required:", Colors.YELLOW)
            print_colored("1. Run: pip install pyinstaller", Colors.WHITE)
            print_colored("2. Then run this script again", Colors.WHITE)
            return False
    
    # Clean previous builds
    for folder in ["dist", "build"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print_success(f"Cleaned {folder} directory")
            except Exception as e:
                print_warning(f"Could not clean {folder}: {e}")
    
    # Remove old spec files
    spec_files = ["career_recommender.spec", "main_lite.spec"]
    for spec_file in spec_files:
        if os.path.exists(spec_file):
            try:
                os.remove(spec_file)
                print_success(f"Removed old {spec_file}")
            except Exception as e:
                print_warning(f"Could not remove {spec_file}: {e}")
    
    print()
    print_step("Starting build process...")
    print_info("This may take several minutes on first run...")
    
    try:
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--windowed", 
            "--name=AI_Career_Recommender",
            "--clean",
            "main_lite.py"
        ]
        
        print_colored(f"🖥️ Running command:", Colors.CYAN)
        print_colored(f"   {' '.join(cmd)}", Colors.WHITE)
        print()
        print_colored("📝 Build output:", Colors.CYAN)
        print_colored("-" * 50, Colors.CYAN)
        
        # Run with real-time output and proper text handling
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Print output in real-time with color coding
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # Color code different types of output
                if "ERROR" in line.upper() or "FAILED" in line.upper():
                    print_colored(line, Colors.RED)
                elif "WARNING" in line.upper():
                    print_colored(line, Colors.YELLOW)
                elif "INFO" in line.upper() or "Building" in line:
                    print_colored(line, Colors.BLUE)
                elif "successfully" in line.lower() or "completed" in line.lower():
                    print_colored(line, Colors.GREEN)
                else:
                    print_colored(line, Colors.WHITE)
        
        process.wait()
        
        print_colored("-" * 50, Colors.CYAN)
        
        if process.returncode == 0:
            print_success("Build completed successfully!")
            
            # Check if executable was created
            exe_path = 'dist/AI_Career_Recommender.exe'
            if os.path.exists(exe_path):
                size_mb = os.path.getsize(exe_path) / (1024 * 1024)
                print_colored(f"📦 Executable location: {os.path.abspath(exe_path)}", Colors.GREEN)
                print_colored(f"📏 File size: {size_mb:.1f} MB", Colors.GREEN)
                
                print()
                print_colored("🎉 Release Information:", Colors.BOLD + Colors.GREEN)
                print_colored("• Cloud AI Analysis: ✅ Included", Colors.GREEN)
                print_colored("• Smart Rule-Based Analysis: ✅ Included", Colors.GREEN)
                print_colored("• 35+ Career Domains: ✅ Included", Colors.GREEN)
                print_colored("• Negative Context Detection: ✅ Included", Colors.GREEN)
                print_colored("• Local BART Model: ❌ Excluded (for size optimization)", Colors.YELLOW)
                print_colored(f"• Target Size: ~{size_mb:.0f}MB vs 2GB+ with full model", Colors.CYAN)
                
                return True
            else:
                print_error("Executable not found in dist/ folder")
                return False
        else:
            print_error("Build failed with errors shown above")
            return False
            
    except FileNotFoundError:
        print_error("PyInstaller command not found")
        print_info("Try installing with: pip install pyinstaller")
        return False
    except Exception as e:
        print_error(f"Build failed with exception: {e}")
        return False

def create_release_package():
    """Create a complete release package with documentation"""
    
    exe_files = [
        'dist/AI_Career_Recommender.exe',
        'dist/main_lite.exe'
    ]
    
    exe_path = None
    for path in exe_files:
        if os.path.exists(path):
            exe_path = path
            break
    
    if not exe_path:
        print("❌ No executable found. Build first.")
        return False
    
    print(f"\n📦 Creating release package...")
    
    # Create release directory
    release_dir = 'AI_Career_Recommender_Release'
    if os.path.exists(release_dir):
        try:
            shutil.rmtree(release_dir)
        except Exception as e:
            print(f"⚠️ Could not clean release directory: {e}")
    
    try:
        os.makedirs(release_dir)
    except Exception as e:
        print(f"❌ Could not create release directory: {e}")
        return False
    
    # Copy executable
    exe_name = os.path.basename(exe_path)
    try:
        shutil.copy2(exe_path, f'{release_dir}/{exe_name}')
        print(f"✅ Copied {exe_name} to release package")
    except Exception as e:
        print(f"❌ Could not copy executable: {e}")
        return False
    
    # Create comprehensive user guide
    user_guide = """
# 🎯 AI Career Recommender - User Guide

## 🎓 BrainWonders Internship Project
**Developed by:** [Your Name]  
**Technologies:** Python, PyQt6, AI/ML, NLP, Cloud Computing  
**Project Type:** AI-Powered Career Guidance System

## 🚀 Quick Start Guide

### Step 1: Launch Application
- Double-click `AI_Career_Recommender.exe`
- No installation required!

### Step 2: Enter Your Profile
Describe your interests, hobbies, and strengths in detail. 

**Example Input:**
```
I love music and am very creative. I enjoy designing things and have a good 
eye for aesthetics. I'm interested in technology and problem-solving but 
I'm not that good at sports or physical activities.
```
### Step 3: Get Career Recommendations
- Receive tailored career suggestions based on your profile.
- Explore detailed information about each suggested career.

### Step 4: Access Additional Resources
- Utilize built-in tools for skill assessment and resume building.
- Access external resources for further learning and development.

## 📁 Included Files
- `AI_Career_Recommender.exe`: Main executable for the application.
- `main_lite.py`: Lightweight Python script for the application core.
- `career_recommender.spec`: PyInstaller spec file for advanced users.

## ⚙️ Technical Details
- **AI Model:** Cloud-based AI analysis with local fallback.
- **Framework:** PyQt6 for the user interface.
- **Packaging:** PyInstaller for creating standalone executables.

## 📞 Support
For issues or feedback, please contact:
- **Email:** support@brainwonders.in
- **Phone:** +91-XXXXXXXXXX

## ©️ License
This project is licensed under the MIT License.
"""
    
    # Write user guide to file
    try:
        with open(f'{release_dir}/USER_GUIDE.md', 'w', encoding='utf-8') as f:
            f.write(user_guide)
        print("📄 Created USER_GUIDE.md")
    except Exception as e:
        print(f"⚠️ Could not create user guide: {e}")
    
    # Copy additional resources if available
    resources = ["README.md", "LICENSE"]
    for resource in resources:
        if os.path.exists(resource):
            try:
                shutil.copy2(resource, release_dir)
                print(f"✅ Copied {resource} to release package")
            except Exception as e:
                print(f"⚠️ Could not copy {resource}: {e}")
    
    print(f"✅ Release package created: {release_dir}/")
    return True

def manual_instructions():
    """Provide manual build instructions with proper formatting"""
    
    print_header("MANUAL BUILD INSTRUCTIONS")
    print()
    print_colored("If automated build fails, follow these steps:", Colors.YELLOW)
    print()
    
    steps = [
    ("Install PyInstaller:", "pip install pyinstaller"),
    ("Simple build command:", "pyinstaller --onefile --windowed --name=AI_Career_Recommender --add-data=.env;. main_lite.py"),
    ("Alternative with exclusions:", "pyinstaller --onefile --windowed --exclude-module=transformers --exclude-module=torch --add-data=.env;. main_lite.py"),
    ("If PyQt6 issues occur:", "pip install --upgrade PyQt6\npyinstaller --onefile --windowed --hidden-import=PyQt6.QtCore --add-data=.env;. main_lite.py"),
    ("Check output:", "Output will be in: dist/AI_Career_Recommender.exe")
]

    
    for i, (description, command) in enumerate(steps, 1):
        print_colored(f"{i}️⃣ {description}", Colors.CYAN)
        if "\n" in command:
            for cmd_line in command.split("\n"):
                print_colored(f"   {cmd_line}", Colors.WHITE)
        else:
            print_colored(f"   {command}", Colors.WHITE)
        print()
    
    print_colored("📁 Current directory contents:", Colors.CYAN)
    try:
        files = os.listdir('.')
        python_files = [f for f in sorted(files) if f.endswith('.py')]
        for file in python_files:
            print_colored(f"   📄 {file}", Colors.WHITE)
    except Exception as e:
        print_colored(f"   Could not list files: {e}", Colors.RED)

def main():
    """Main function with proper error handling and colored output"""
    
    # Enable color support on Windows
    if os.name == 'nt':
        try:
            # Enable ANSI escape sequences on Windows 10+
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            # Fallback - disable colors
            for attr in dir(Colors):
                if not attr.startswith('_'):
                    setattr(Colors, attr, '')
    
    print_header("AI Career Recommender - Build System")
    print_colored("🎓 BrainWonders Internship Project", Colors.BOLD + Colors.CYAN)
    print()
    
    # Check current directory
    current_dir = os.getcwd()
    print_colored(f"📁 Current directory: {current_dir}", Colors.BLUE)
    
    # Check for required files
    required_files = ['main_lite.py']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print_success(f"Found: {file}")
        else:
            print_error(f"Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print()
        print_warning(f"Missing required files: {missing_files}")
        print_info("Please ensure main_lite.py exists before building.")
        return
    
    # Get user choice with colored prompts
    print()
    print_colored("Choose build option:", Colors.BOLD + Colors.CYAN)
    print_colored("1. Build executable only", Colors.WHITE)
    print_colored("2. Build + create release package", Colors.WHITE)
    print_colored("3. Show manual instructions", Colors.WHITE)
    
    try:
        print_colored("Enter choice (1, 2, or 3): ", Colors.YELLOW, end="")
        choice = input()
    except KeyboardInterrupt:
        print()
        print_colored("👋 Build cancelled by user", Colors.YELLOW)
        return
    except Exception as e:
        print()
        print_error(f"Input error: {e}")
        return
    
    print()
    
    if choice == "1":
        print_step("Building executable...")
        success = create_executable()
        if not success:
            print()
            print_colored("📋 Showing manual instructions as fallback...", Colors.YELLOW)
            manual_instructions()
    elif choice == "2":
        print_step("Building executable and creating release package...")
        success = create_executable()
        if success:
            create_release_package()
        else:
            print()
            print_colored("📋 Showing manual instructions as fallback...", Colors.YELLOW)
            manual_instructions()
    elif choice == "3":
        manual_instructions()
    else:
        print_error("Invalid choice. Please run again and enter 1, 2, or 3.")
    
    print()
    print_colored("🎉 Build Process Complete!", Colors.BOLD + Colors.GREEN)
    print_colored("🎓 BrainWonders Internship Project - AI Career Recommender", Colors.CYAN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("👋 Build process interrupted by user", Colors.YELLOW)
    except Exception as e:
        print()
        print_colored(f"💥 Unexpected error: {e}", Colors.RED)
        print_colored("📋 Try running manual build commands instead", Colors.YELLOW)
        manual_instructions()