# api_launcher.py
"""
Simple launcher for any Python library API
Just run: python api_launcher.py <library_name>
"""

import os
import sys
import yaml
import json
import argparse
from typing import Dict, Any


class SimpleAPILauncher:
    """Dead simple API launcher for any library"""
    
    # One-line configurations for common libraries
    SIMPLE_CONFIGS = {
        # Computer Vision
        'yolo': {
            'install': 'pip install ultralytics',
            'import': 'from ultralytics import YOLO',
            'init': 'model = YOLO("yolov8n.pt")',
            'endpoints': {
                'detect': 'lambda img: model(img)',
                'track': 'lambda img: model.track(img, persist=True)'
            }
        },
        
        # LLMs
        'gpt2': {
            'install': 'pip install transformers torch',
            'import': 'from transformers import pipeline',
            'init': 'model = pipeline("text-generation", model="gpt2")',
            'endpoints': {
                'generate': 'lambda text: model(text, max_length=100)'
            }
        },
        
        'phi2': {
            'install': 'pip install transformers torch',
            'import': 'from transformers import AutoModelForCausalLM, AutoTokenizer',
            'init': '''
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
            ''',
            'endpoints': {
                'generate': '''lambda prompt: tokenizer.decode(
                    model.generate(
                        tokenizer(prompt, return_tensors="pt").input_ids,
                        max_length=100
                    )[0]
                )'''
            }
        },
        
        # MLX (Apple Silicon)
        'mlx_whisper': {
            'install': 'pip install mlx-whisper',
            'import': 'import mlx_whisper',
            'init': '# MLX Whisper loads models on demand',
            'endpoints': {
                'transcribe': 'lambda audio: mlx_whisper.transcribe(audio, path_or_hf_repo="tiny")'
            }
        },
        
        'mlx_stable_diffusion': {
            'install': 'pip install mlx-stable-diffusion',
            'import': 'from mlx_stable_diffusion import StableDiffusion',
            'init': 'model = StableDiffusion()',
            'endpoints': {
                'generate': 'lambda prompt: model.generate_image(prompt)'
            }
        },
        
        # Data Science
        'pandas': {
            'install': 'pip install pandas',
            'import': 'import pandas as pd',
            'init': '# Pandas is ready',
            'endpoints': {
                'read_csv': 'pd.read_csv',
                'read_excel': 'pd.read_excel',
                'describe': 'lambda df: df.describe().to_dict()'
            }
        },
        
        'sklearn': {
            'install': 'pip install scikit-learn',
            'import': 'from sklearn.ensemble import RandomForestClassifier',
            'init': 'model = RandomForestClassifier()',
            'endpoints': {
                'fit': 'lambda X, y: model.fit(X, y)',
                'predict': 'lambda X: model.predict(X).tolist()'
            }
        },
        
        # Image Processing
        'opencv': {
            'install': 'pip install opencv-python',
            'import': 'import cv2',
            'init': '# OpenCV ready',
            'endpoints': {
                'resize': 'lambda img, size: cv2.resize(img, size)',
                'blur': 'lambda img: cv2.GaussianBlur(img, (5,5), 0)',
                'edge_detect': 'lambda img: cv2.Canny(img, 100, 200)'
            }
        },
        
        'pillow': {
            'install': 'pip install Pillow',
            'import': 'from PIL import Image, ImageFilter',
            'init': '# PIL ready',
            'endpoints': {
                'open': 'Image.open',
                'resize': 'lambda img, size: img.resize(size)',
                'filter': 'lambda img, f: img.filter(getattr(ImageFilter, f))'
            }
        }
    }
    
    @classmethod
    def create_simple_api(cls, library: str, port: int = 8000):
        """Create API from simple config"""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
        
        if library not in cls.SIMPLE_CONFIGS:
            print(f"❌ Unknown library: {library}")
            print(f"Available: {', '.join(cls.SIMPLE_CONFIGS.keys())}")
            return
            
        config = cls.SIMPLE_CONFIGS[library]
        
        # Check if library is installed
        try:
            exec(config['import'])
        except ImportError:
            print(f"📦 {library} not installed. Install with:")
            print(f"   {config['install']}")
            return
            
        # Create FastAPI app
        app = FastAPI(
            title=f"{library.upper()} API",
            description=f"Auto-generated API for {library}"
        )
        
        # Initialize model/library
        exec(config['import'], globals())
        exec(config['init'], globals())
        
        # Create endpoints
        for endpoint_name, endpoint_code in config['endpoints'].items():
            # Create the function
            if endpoint_code.startswith('lambda'):
                func = eval(endpoint_code)
            else:
                func = eval(endpoint_code)
                
            # Create FastAPI endpoint
            def create_endpoint(name, fn):
                def endpoint(**kwargs):
                    try:
                        result = fn(**kwargs)
                        
                        # Convert numpy arrays and other types
                        if hasattr(result, 'tolist'):
                            result = result.tolist()
                        elif hasattr(result, '__dict__'):
                            result = result.__dict__
                            
                        return {"result": result, "success": True}
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=str(e))
                        
                endpoint.__name__ = f"{name}_endpoint"
                return endpoint
                
            # Add to FastAPI
            app.post(f"/{endpoint_name}")(create_endpoint(endpoint_name, func))
            
        # Add health check
        @app.get("/")
        def root():
            return {
                "library": library,
                "endpoints": [f"/{name}" for name in config['endpoints'].keys()],
                "status": "ready"
            }
            
        # Run the server
        print(f"\n🚀 Starting {library.upper()} API")
        print(f"📍 URL: http://localhost:{port}")
        print(f"📚 Docs: http://localhost:{port}/docs")
        print(f"\nEndpoints:")
        for ep in config['endpoints'].keys():
            print(f"  POST /{ep}")
        print("\nPress Ctrl+C to stop\n")
        
        uvicorn.run(app, host="0.0.0.0", port=port)


# ============= Even Simpler: One Command =============

def one_command_api():
    """
    Super simple API creation
    Usage: python -c "from api_launcher import *; api('yolo')"
    """
    def api(library: str, port: int = 8000):
        SimpleAPILauncher.create_simple_api(library, port)
    
    return api

# Make it available globally
api = one_command_api()


# ============= Auto-installer =============

class AutoInstaller:
    """Automatically install and configure libraries"""
    
    @staticmethod
    def setup_library(library: str):
        """Auto-install and configure a library"""
        configs = SimpleAPILauncher.SIMPLE_CONFIGS
        
        if library not in configs:
            print(f"Unknown library: {library}")
            return False
            
        config = configs[library]
        
        # Try import
        try:
            exec(config['import'])
            print(f"✅ {library} is already installed")
            return True
        except ImportError:
            print(f"📦 Installing {library}...")
            os.system(config['install'])
            
            # Verify installation
            try:
                exec(config['import'])
                print(f"✅ {library} installed successfully")
                return True
            except ImportError:
                print(f"❌ Failed to install {library}")
                return False


# ============= CLI =============

def main():
    parser = argparse.ArgumentParser(
        description="Universal API Launcher - Turn any Python library into an API instantly"
    )
    parser.add_argument(
        "library",
        help="Library name (yolo, gpt2, pandas, sklearn, opencv, etc.)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run the API on"
    )
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="Auto-install the library if not found"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available libraries"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📚 Available Libraries:")
        for lib, config in SimpleAPILauncher.SIMPLE_CONFIGS.items():
            print(f"\n{lib}:")
            print(f"  Install: {config['install']}")
            print(f"  Endpoints: {', '.join(config['endpoints'].keys())}")
        return
        
    if args.install:
        AutoInstaller.setup_library(args.library)
        
    SimpleAPILauncher.create_simple_api(args.library, args.port)


if __name__ == "__main__":
    main()


# ============= Ultra Simple Usage =============
"""
USAGE EXAMPLES:

1. One-liner to start any API:
   python api_launcher.py yolo

2. With custom port:
   python api_launcher.py gpt2 --port 8080

3. Auto-install if needed:
   python api_launcher.py pandas --install

4. List available libraries:
   python api_launcher.py --list

5. In Python script:
   from api_launcher import api
   api('yolo', port=8001)

6. Test with curl:
   curl -X POST http://localhost:8000/detect \
     -H "Content-Type: application/json" \
     -d '{"img": "path/to/image.jpg"}'
"""
