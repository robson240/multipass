# universal_api_wrapper.py
"""
Universal API Wrapper - Automatically wrap any Python library with a robust REST API
Works with YOLO, LLMs, Ultralytics, Streamlit, MLX, and any other Python library
"""

import asyncio
import base64
import importlib
import inspect
import io
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Union, Type

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, create_model
import uvicorn

logger = logging.getLogger(__name__)


# ============= Universal Service Discovery =============

class UniversalServiceDiscovery:
    """Automatically discovers and wraps ANY library's services"""
    
    def __init__(self, module_name: str, module_config: Dict[str, Any] = None):
        self.module_name = module_name
        self.module_config = module_config or {}
        self.module = None
        self.services = {}
        self.models = {}
        self.metrics = {}
        self._initialize_module()
        
    def _initialize_module(self):
        """Import and initialize the target module"""
        try:
            # Import the module
            self.module = importlib.import_module(self.module_name)
            logger.info(f"Successfully imported {self.module_name}")
            
            # Run custom initialization if provided
            if 'init_function' in self.module_config:
                init_fn = getattr(self.module, self.module_config['init_function'])
                init_args = self.module_config.get('init_args', {})
                self.initialized_object = init_fn(**init_args)
                logger.info(f"Initialized {self.module_name} with custom function")
            
            # Discover services
            self._discover_services()
            
        except ImportError as e:
            logger.error(f"Failed to import {self.module_name}: {e}")
            raise
            
    def _discover_services(self):
        """Discover all callable services in the module"""
        # Get discovery targets
        if hasattr(self, 'initialized_object'):
            discovery_targets = [
                (self.module_name, self.module),
                ('initialized_object', self.initialized_object)
            ]
        else:
            discovery_targets = [(self.module_name, self.module)]
            
        for target_name, target in discovery_targets:
            for name, obj in inspect.getmembers(target):
                if name.startswith('_'):
                    continue
                    
                if callable(obj) or inspect.isclass(obj):
                    service_key = f"{target_name}.{name}" if target_name != self.module_name else name
                    self.services[service_key] = {
                        'name': name,
                        'object': obj,
                        'type': 'class' if inspect.isclass(obj) else 'function',
                        'signature': self._get_signature(obj),
                        'docstring': inspect.getdoc(obj),
                        'target': target_name
                    }
                    self.metrics[service_key] = {
                        'call_count': 0,
                        'error_count': 0,
                        'total_time': 0
                    }
                    
    def _get_signature(self, obj: Any) -> Dict[str, Any]:
        """Extract function/method signature"""
        try:
            sig = inspect.signature(obj)
            params = {}
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': repr(param.default) if param.default != inspect.Parameter.empty else None,
                    'kind': str(param.kind)
                }
                params[param_name] = param_info
                
            return {
                'parameters': params,
                'return_type': str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else 'Any'
            }
        except:
            return {'parameters': {}, 'return_type': 'Any'}
            
    def call_service(self, service_name: str, *args, **kwargs) -> Any:
        """Call a discovered service with metrics"""
        if service_name not in self.services:
            raise ValueError(f"Service '{service_name}' not found")
            
        service = self.services[service_name]['object']
        metrics = self.metrics[service_name]
        
        start_time = time.time()
        try:
            result = service(*args, **kwargs)
            metrics['call_count'] += 1
            return result
        except Exception as e:
            metrics['error_count'] += 1
            raise
        finally:
            metrics['total_time'] += time.time() - start_time


# ============= Library-Specific Adapters =============

class YOLOAdapter:
    """Adapter for YOLO/Ultralytics models"""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.model_name = model_name
        
    def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run object detection"""
        results = self.model(image, **kwargs)
        
        # Convert results to serializable format
        detections = []
        for r in results:
            if r.boxes:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': r.names[int(cls)]
                    })
                    
        return {
            'detections': detections,
            'count': len(detections),
            'model': self.model_name
        }
        
    def segment(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run instance segmentation"""
        results = self.model(image, task='segment', **kwargs)
        # Process segmentation results
        return {'segments': []}  # Simplified
        
    def track(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run object tracking"""
        results = self.model.track(image, persist=True, **kwargs)
        # Process tracking results
        return {'tracks': []}  # Simplified


class LLMAdapter:
    """Adapter for various LLM libraries"""
    
    def __init__(self, provider: str = "transformers", model_name: str = "gpt2"):
        self.provider = provider
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the LLM based on provider"""
        if self.provider == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
        elif self.provider == "openai":
            import openai
            self.model = openai
            
        elif self.provider == "anthropic":
            import anthropic
            self.model = anthropic.Anthropic()
            
        elif self.provider == "mlx_lm":
            import mlx_lm
            self.model, self.tokenizer = mlx_lm.load(self.model_name)
            
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text based on prompt"""
        if self.provider == "transformers":
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, **kwargs)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        elif self.provider == "openai":
            response = self.model.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            text = response.choices[0].message.content
            
        elif self.provider == "mlx_lm":
            text = mlx_lm.generate(self.model, self.tokenizer, prompt, **kwargs)
            
        else:
            text = f"Provider {self.provider} not implemented"
            
        return {
            'text': text,
            'model': self.model_name,
            'provider': self.provider,
            'prompt_tokens': len(prompt.split()),
            'completion_tokens': len(text.split())
        }
        
    def embed(self, text: str) -> List[float]:
        """Generate embeddings"""
        # Implementation depends on provider
        return []


class StreamlitAdapter:
    """Adapter for Streamlit applications"""
    
    def __init__(self, app_path: str):
        self.app_path = app_path
        self.process = None
        
    def start_app(self, port: int = 8501) -> Dict[str, Any]:
        """Start Streamlit app in subprocess"""
        import subprocess
        
        self.process = subprocess.Popen(
            ["streamlit", "run", self.app_path, "--server.port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return {
            'status': 'started',
            'port': port,
            'url': f"http://localhost:{port}",
            'pid': self.process.pid
        }
        
    def stop_app(self) -> Dict[str, Any]:
        """Stop Streamlit app"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            return {'status': 'stopped'}
        return {'status': 'not_running'}
        
    def get_app_state(self) -> Dict[str, Any]:
        """Get Streamlit app state (if using session state API)"""
        # This would require integration with Streamlit's session state
        return {}


# ============= Universal API Factory =============

class UniversalAPIFactory:
    """Factory for creating APIs for any library"""
    
    # Preset configurations for popular libraries
    PRESETS = {
        'yolo': {
            'module': 'ultralytics',
            'adapter_class': YOLOAdapter,
            'init_args': {'model_name': 'yolov8n.pt'}
        },
        'transformers': {
            'module': 'transformers',
            'adapter_class': LLMAdapter,
            'init_args': {'provider': 'transformers', 'model_name': 'gpt2'}
        },
        'mlx_whisper': {
            'module': 'mlx_whisper',
            'endpoints': ['transcribe', 'load_model']
        },
        'opencv': {
            'module': 'cv2',
            'endpoints': ['imread', 'imwrite', 'resize', 'cvtColor']
        },
        'pandas': {
            'module': 'pandas',
            'endpoints': ['read_csv', 'read_excel', 'DataFrame']
        },
        'scikit-learn': {
            'module': 'sklearn',
            'submodules': ['ensemble', 'svm', 'neural_network']
        }
    }
    
    @classmethod
    def create_api(
        cls,
        library_name: str,
        custom_config: Dict[str, Any] = None
    ) -> FastAPI:
        """Create a FastAPI app for any library"""
        
        # Get configuration
        config = cls.PRESETS.get(library_name, {})
        if custom_config:
            config.update(custom_config)
            
        # Create FastAPI app
        app = FastAPI(
            title=f"{library_name.upper()} API",
            description=f"Auto-generated API for {library_name}",
            version="1.0.0"
        )
        
        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Initialize adapter or discovery
        if 'adapter_class' in config:
            adapter = config['adapter_class'](**config.get('init_args', {}))
            app.state.adapter = adapter
        else:
            discovery = UniversalServiceDiscovery(
                config.get('module', library_name),
                config
            )
            app.state.discovery = discovery
            
        # Create endpoints
        cls._create_endpoints(app, library_name, config)
        
        return app
        
    @classmethod
    def _create_endpoints(cls, app: FastAPI, library_name: str, config: Dict):
        """Create API endpoints based on configuration"""
        
        @app.get("/")
        async def root():
            return {
                "library": library_name,
                "status": "operational",
                "endpoints": [route.path for route in app.routes]
            }
            
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "library": library_name,
                "timestamp": time.time()
            }
            
        if hasattr(app.state, 'adapter'):
            # Create adapter-specific endpoints
            adapter = app.state.adapter
            
            for method_name in dir(adapter):
                if not method_name.startswith('_'):
                    method = getattr(adapter, method_name)
                    if callable(method):
                        cls._create_endpoint_for_method(
                            app, method_name, method, library_name
                        )
                        
        elif hasattr(app.state, 'discovery'):
            # Create discovery-based endpoints
            discovery = app.state.discovery
            
            @app.get("/services")
            async def list_services():
                return {
                    service_name: {
                        'type': info['type'],
                        'signature': info['signature'],
                        'docstring': info['docstring']
                    }
                    for service_name, info in discovery.services.items()
                }
                
            @app.post("/call/{service_name}")
            async def call_service(service_name: str, request: Dict[str, Any]):
                try:
                    args = request.get('args', [])
                    kwargs = request.get('kwargs', {})
                    
                    result = discovery.call_service(service_name, *args, **kwargs)
                    
                    # Handle different result types
                    if isinstance(result, np.ndarray):
                        result = result.tolist()
                    elif hasattr(result, '__dict__'):
                        result = result.__dict__
                        
                    return {'result': result, 'service': service_name}
                    
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
                    
    @classmethod
    def _create_endpoint_for_method(cls, app, method_name, method, library_name):
        """Create an endpoint for a specific method"""
        
        # Dynamic endpoint creation
        @app.post(f"/{method_name}")
        async def endpoint(request: Dict[str, Any]):
            try:
                # Handle image inputs
                if 'image_base64' in request:
                    image_data = base64.b64decode(request['image_base64'])
                    image = np.frombuffer(image_data, dtype=np.uint8)
                    # Reshape if needed (assuming it's an image)
                    request['image'] = image
                    del request['image_base64']
                    
                result = method(**request)
                return result
                
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        # Update endpoint name and docs
        endpoint.__name__ = f"{method_name}_endpoint"
        endpoint.__doc__ = f"Call {library_name}.{method_name}"


# ============= Usage Examples =============

def create_yolo_api():
    """Example: Create YOLO API"""
    return UniversalAPIFactory.create_api('yolo', {
        'init_args': {'model_name': 'yolov8x.pt'}
    })


def create_llm_api():
    """Example: Create LLM API"""
    return UniversalAPIFactory.create_api('transformers', {
        'init_args': {
            'provider': 'transformers',
            'model_name': 'microsoft/phi-2'
        }
    })


def create_custom_api():
    """Example: Create API for any custom library"""
    return UniversalAPIFactory.create_api('my_custom_lib', {
        'module': 'my_package.my_module',
        'init_function': 'initialize',
        'init_args': {'config': 'production'},
        'endpoints': ['process', 'analyze', 'export']
    })


# ============= CLI Interface =============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Library API Wrapper")
    parser.add_argument("library", help="Library name or preset (yolo, transformers, etc.)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", type=str, help="JSON config file path")
    parser.add_argument("--model", type=str, help="Model name/path for ML libraries")
    
    args = parser.parse_args()
    
    # Load custom config if provided
    custom_config = {}
    if args.config:
        with open(args.config) as f:
            custom_config = json.load(f)
            
    if args.model:
        custom_config.setdefault('init_args', {})['model_name'] = args.model
        
    # Create and run API
    app = UniversalAPIFactory.create_api(args.library, custom_config)
    
    print(f"\n🚀 Starting {args.library.upper()} API on http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔧 Interactive API: http://{args.host}:{args.port}/redoc\n")
    
    uvicorn.run(app, host=args.host, port=args.port)
