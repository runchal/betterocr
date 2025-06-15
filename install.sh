#!/bin/bash
# BetterOCR Installation Script
# Installs OCR engines and dependencies system-wide

echo "BetterOCR System-Wide Installation"
echo "=================================="
echo "This will install OCR engines and Python packages system-wide"
echo "so they're available to all your applications."
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    OS="linux"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
echo ""
echo "Step 1: Installing system dependencies..."
echo "-----------------------------------------"

if [ "$OS" == "macos" ]; then
    # Check for Homebrew
    if ! command_exists brew; then
        echo "Homebrew not found. Please install it first:"
        echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    echo "Installing Tesseract..."
    brew install tesseract
    
    echo "Installing Tesseract language packs..."
    brew install tesseract-lang
    
    echo "Installing poppler for PDF support..."
    brew install poppler
    
    echo "Installing other dependencies..."
    brew install libmagic
    
elif [ "$OS" == "linux" ]; then
    echo "Installing Tesseract..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-all
    
    echo "Installing poppler for PDF support..."
    sudo apt-get install -y poppler-utils
    
    echo "Installing other dependencies..."
    sudo apt-get install -y libmagic1
fi

# Install Python packages system-wide
echo ""
echo "Step 2: Installing Python packages..."
echo "-------------------------------------"
echo "Note: This will install packages system-wide."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Core OCR packages
echo "Installing Tesseract Python wrapper..."
pip3 install pytesseract

echo "Installing EasyOCR (this may take a while)..."
pip3 install easyocr

echo "Installing PaddleOCR..."
# PaddlePaddle has different packages for different systems
if [ "$OS" == "macos" ]; then
    if [[ $(uname -m) == 'arm64' ]]; then
        echo "Installing PaddlePaddle for Apple Silicon..."
        pip3 install paddlepaddle==2.5.1
    else
        echo "Installing PaddlePaddle for Intel Mac..."
        pip3 install paddlepaddle==2.5.1
    fi
else
    echo "Installing PaddlePaddle for Linux..."
    pip3 install paddlepaddle
fi
pip3 install paddleocr

# Image processing packages
echo "Installing image processing libraries..."
pip3 install Pillow opencv-python pdf2image

# ML packages for computer vision
echo "Installing computer vision packages..."
pip3 install torch torchvision transformers

# Utility packages
echo "Installing utility packages..."
pip3 install numpy pandas python-magic reportlab

# Verify installations
echo ""
echo "Step 3: Verifying installations..."
echo "-----------------------------------"

python3 << EOF
import sys
print("Python version:", sys.version)
print("\nChecking installed packages:")

packages = [
    ("pytesseract", "Tesseract Python"),
    ("easyocr", "EasyOCR"),
    ("paddleocr", "PaddleOCR"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("pdf2image", "PDF2Image"),
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
]

for module_name, display_name in packages:
    try:
        if module_name == "cv2":
            import cv2
        elif module_name == "PIL":
            from PIL import Image
        else:
            __import__(module_name)
        print(f"✓ {display_name} installed successfully")
    except ImportError:
        print(f"✗ {display_name} installation failed")
EOF

echo ""
echo "Installation complete!"
echo ""
echo "To test the installation, run:"
echo "  python3 test_basic.py"
echo ""
echo "To process a document:"
echo "  python3 betterocr.py samples/your_document.pdf"