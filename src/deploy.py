import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any
import docker
from fastapi import FastAPI, HTTPException
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def export_to_onnx(self, output_path: str):
        """Export PyTorch model to ONNX format."""
        model = TimeSeriesDBN.load_from_checkpoint(self.model_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, self.config['sequence_length'], 
                                self.config['input_size'])
        
        # Export model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Model exported to {output_path}")
    
    def create_docker_image(self, dockerfile_path: str):
        """Create Docker image for deployment."""
        client = docker.from_env()
        
        # Build image
        image, _ = client.images.build(
            path=str(Path(dockerfile_path).parent),
            tag="anomaly-detector:latest"
        )
        
        logger.info(f"Docker image created: {image.tags}")
    
    def create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application for model serving."""
        app = FastAPI(title="Anomaly Detection API")
        
        # Load ONNX model
        session = onnxruntime.InferenceSession(
            self.config['onnx_model_path'],
            providers=['CPUExecutionProvider']
        )
        
        @app.post("/predict")
        async def predict(data: Dict[str, Any]):
            try:
                # Preprocess input
                input_data = self._preprocess_input(data)
                
                # Run inference
                output = session.run(
                    ['output'],
                    {'input': input_data}
                )
                
                # Postprocess output
                result = self._postprocess_output(output[0])
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    def _preprocess_input(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data for model inference."""
        # Implement preprocessing logic
        pass
    
    def _postprocess_output(self, output: np.ndarray) -> Dict[str, Any]:
        """Postprocess model output."""
        # Implement postprocessing logic
        pass
    
    def deploy(self, host: str = "0.0.0.0", port: int = 8000):
        """Deploy the model as a REST API."""
        app = self.create_fastapi_app()
        uvicorn.run(app, host=host, port=port)

def main():
    # Example usage
    deployer = ModelDeployer(
        model_path="models/anomaly_detector.pth",
        config_path="config/deployment_config.json"
    )
    
    # Export to ONNX
    deployer.export_to_onnx("models/anomaly_detector.onnx")
    
    # Create Docker image
    deployer.create_docker_image("Dockerfile")
    
    # Deploy API
    deployer.deploy()

if __name__ == "__main__":
    main() 