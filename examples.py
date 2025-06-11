# examples.py
"""
Examples of using the Universal API Wrapper with different libraries
"""

import asyncio
import base64
import numpy as np
from pathlib import Path


# ============= 1. YOLO Object Detection API =============

async def yolo_example():
    """Example: Object detection with YOLO"""
    from universal_api_wrapper import UniversalAPIFactory
    
    # Create YOLO API
    app = UniversalAPIFactory.create_api('yolo')
    
    # In practice, you'd run this as a server
    # uvicorn.run(app, host="0.0.0.0", port=8001)
    
    # Client usage
    import httpx
    
    # Load and encode image
    with open("test_image.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    async with httpx.AsyncClient() as client:
        # Object detection
        response = await client.post(
            "http://localhost:8001/detect",
            json={
                "image_base64": image_base64,
                "conf": 0.25,  # confidence threshold
                "iou": 0.45    # IoU threshold
            }
        )
        
        result = response.json()
        print(f"Detected {result['count']} objects:")
        for det in result['detections']:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")


# ============= 2. LLM Text Generation API =============

async def llm_example():
    """Example: Text generation with various LLMs"""
    from universal_api_wrapper import UniversalAPIFactory
    
    # Create LLM API with custom model
    app = UniversalAPIFactory.create_api('transformers', {
        'init_args': {
            'provider': 'transformers',
            'model_name': 'gpt2'  # or 'microsoft/phi-2', 'meta-llama/Llama-2-7b'
        }
    })
    
    # Client usage
    import httpx
    
    async with httpx.AsyncClient() as client:
        # Generate text
        response = await client.post(
            "http://localhost:8002/generate",
            json={
                "prompt": "The future of AI is",
                "max_length": 100,
                "temperature": 0.8,
                "top_p": 0.9
            }
        )
        
        result = response.json()
        print(f"Generated text: {result['text']}")
        print(f"Tokens used: {result['prompt_tokens']} + {result['completion_tokens']}")


# ============= 3. Streamlit App Manager =============

def streamlit_example():
    """Example: Managing Streamlit apps via API"""
    from universal_api_wrapper import UniversalAPIFactory
    
    # Create Streamlit manager API
    app = UniversalAPIFactory.create_api('streamlit', {
        'adapter_class': 'StreamlitAdapter',
        'init_args': {
            'app_path': 'my_dashboard.py'
        }
    })
    
    # This would start/stop Streamlit apps programmatically
    # POST /start_app -> Starts the Streamlit app
    # POST /stop_app -> Stops the app
    # GET /get_app_state -> Gets app state


# ============= 4. Computer Vision Pipeline =============

async def opencv_pipeline_example():
    """Example: Computer vision pipeline with OpenCV"""
    from universal_api_wrapper import UniversalAPIFactory
    
    # Create OpenCV API
    app = UniversalAPIFactory.create_api('opencv')
    
    import httpx
    
    async with httpx.AsyncClient() as client:
        # Chain multiple operations
        operations = [
            {
                "service": "cvtColor",
                "args": [],
                "kwargs": {"code": "COLOR_BGR2GRAY"}
            },
            {
                "service": "GaussianBlur",
                "args": [],
                "kwargs": {"ksize": [5, 5], "sigmaX": 0}
            },
            {
                "service": "Canny",
                "args": [],
                "kwargs": {"threshold1": 100, "threshold2": 200}
            }
        ]
        
        # Process image through pipeline
        response = await client.post(
            "http://localhost:8003/pipeline",
            json={
                "image_base64": image_base64,
                "operations": operations
            }
        )


# ============= 5. Data Processing with Pandas =============

async def pandas_example():
    """Example: Data processing with Pandas"""
    from universal_api_wrapper import UniversalAPIFactory
    
    # Create Pandas API
    app = UniversalAPIFactory.create_api('pandas')
    
    import httpx
    
    async with httpx.AsyncClient() as client:
        # Upload and process CSV
        files = {'file': open('data.csv', 'rb')}
        
        # Read CSV
        response = await client.post(
            "http://localhost:8004/read_csv",
            files=files
        )
        
        # Get statistics
        response = await client.post(
            "http://localhost:8004/call/DataFrame.describe",
            json={"args": [], "kwargs": {}}
        )
        
        stats = response.json()
        print("Data statistics:", stats)


# ============= 6. Universal Client Library =============

class UniversalAPIClient:
    """Universal client for any wrapped library"""
    
    def __init__(self, base_url: str, library_name: str):
        self.base_url = base_url
        self.library_name = library_name
        
    async def discover_services(self):
        """Discover available services"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/services")
            return response.json()
            
    async def call_service(self, service_name: str, *args, **kwargs):
        """Call any service dynamically"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/call/{service_name}",
                json={"args": list(args), "kwargs": kwargs}
            )
            return response.json()['result']
            
    async def __getattr__(self, name):
        """Dynamic method creation"""
        async def method(*args, **kwargs):
            return await self.call_service(name, *args, **kwargs)
        return method


# ============= 7. Multi-Library Orchestration =============

class MLPipeline:
    """Orchestrate multiple library APIs"""
    
    def __init__(self):
        self.yolo_client = UniversalAPIClient("http://localhost:8001", "yolo")
        self.llm_client = UniversalAPIClient("http://localhost:8002", "llm")
        self.cv_client = UniversalAPIClient("http://localhost:8003", "opencv")
        
    async def analyze_image_with_description(self, image_path: str):
        """Complete pipeline: detect objects and generate descriptions"""
        
        # 1. Detect objects with YOLO
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode()
            
        detections = await self.yolo_client.detect(
            image_base64=image_base64,
            conf=0.5
        )
        
        # 2. Generate description with LLM
        objects = [d['class_name'] for d in detections['detections']]
        prompt = f"Describe a scene containing: {', '.join(objects)}"
        
        description = await self.llm_client.generate(
            prompt=prompt,
            max_length=150
        )
        
        return {
            'objects': objects,
            'count': len(objects),
            'description': description['text'],
            'detections': detections['detections']
        }


# ============= 8. Quick Start Script =============

def quick_start():
    """Quick start script to launch multiple APIs"""
    import subprocess
    import time
    
    libraries = [
        ('yolo', 8001, 'yolov8n.pt'),
        ('transformers', 8002, 'gpt2'),
        ('opencv', 8003, None),
        ('pandas', 8004, None)
    ]
    
    processes = []
    
    for lib, port, model in libraries:
        cmd = ['python', 'universal_api_wrapper.py', lib, '--port', str(port)]
        if model:
            cmd.extend(['--model', model])
            
        print(f"Starting {lib} API on port {port}...")
        proc = subprocess.Popen(cmd)
        processes.append(proc)
        time.sleep(2)  # Give it time to start
        
    print("\nAll APIs started! Press Ctrl+C to stop all.")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all APIs...")
        for proc in processes:
            proc.terminate()
            proc.wait()


# ============= 9. Docker Compose for All Services =============

DOCKER_COMPOSE_ALL = """
version: '3.8'

services:
  yolo-api:
    build: .
    command: python universal_api_wrapper.py yolo --host 0.0.0.0 --port 8000 --model yolov8x.pt
    ports:
      - "8001:8000"
    volumes:
      - ./models:/models
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  llm-api:
    build: .
    command: python universal_api_wrapper.py transformers --host 0.0.0.0 --port 8000 --model microsoft/phi-2
    ports:
      - "8002:8000"
    volumes:
      - ./models:/models
    environment:
      - TRANSFORMERS_CACHE=/models

  opencv-api:
    build: .
    command: python universal_api_wrapper.py opencv --host 0.0.0.0 --port 8000
    ports:
      - "8003:8000"
    volumes:
      - ./data:/data

  streamlit-manager:
    build: .
    command: python universal_api_wrapper.py streamlit --host 0.0.0.0 --port 8000
    ports:
      - "8004:8000"
      - "8501-8510:8501-8510"  # Range for Streamlit apps
    volumes:
      - ./apps:/apps

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-universal.conf:/etc/nginx/nginx.conf
    depends_on:
      - yolo-api
      - llm-api
      - opencv-api
"""

# ============= 10. Testing Suite =============

async def test_all_apis():
    """Test all deployed APIs"""
    import httpx
    
    apis = {
        'yolo': 'http://localhost:8001',
        'llm': 'http://localhost:8002',
        'opencv': 'http://localhost:8003',
        'pandas': 'http://localhost:8004'
    }
    
    results = {}
    
    async with httpx.AsyncClient() as client:
        for name, url in apis.items():
            try:
                # Health check
                response = await client.get(f"{url}/health")
                results[name] = {
                    'status': response.json()['status'],
                    'available': True
                }
                
                # Get services
                response = await client.get(f"{url}/services")
                services = response.json()
                results[name]['services'] = len(services)
                
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'available': False,
                    'error': str(e)
                }
                
    # Print results
    print("\n=== API Status Report ===")
    for name, info in results.items():
        status = "✅" if info['available'] else "❌"
        print(f"{status} {name}: {info['status']}")
        if 'services' in info:
            print(f"   Services: {info['services']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == "yolo":
            asyncio.run(yolo_example())
        elif example == "llm":
            asyncio.run(llm_example())
        elif example == "pipeline":
            pipeline = MLPipeline()
            result = asyncio.run(
                pipeline.analyze_image_with_description("test.jpg")
            )
            print(result)
        elif example == "test":
            asyncio.run(test_all_apis())
        elif example == "quickstart":
            quick_start()
    else:
        print("Usage: python examples.py [yolo|llm|pipeline|test|quickstart]")
