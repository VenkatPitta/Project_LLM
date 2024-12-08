# First, install the required packages in Terminal:
"""
# Create a new conda environment
conda create -n stable-diffusion python=3.10
conda activate stable-diffusion

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install Diffusers and other requirements
pip install diffusers transformers accelerate safetensors
"""

import torch
from diffusers import StableDiffusionPipeline
import json
import os
from tqdm import tqdm
import time
from PIL import Image

class LocalImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", output_dir="generated_images"):
        """
        Initialize the local image generator
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            output_dir: Directory to save generated images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
        self.pipe.to(self.device)
        
        # Enable attention slicing for lower memory usage
        self.pipe.enable_attention_slicing()
        
    def generate_single_image(self, description: str, index: int) -> dict:
        """
        Generate a single image from a description
        
        Args:
            description: Text description of the image to generate
            index: Index number for the image
            
        Returns:
            Dict containing image metadata
        """
        try:
            # Generate the image
            image = self.pipe(
                prompt=description,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            # Save the image
            image_path = os.path.join(self.output_dir, f"image_{index:04d}.png")
            image.save(image_path)
            
            return {
                "image_path": image_path,
                "description": description,
                "generation_params": {
                    "steps": 30,
                    "guidance_scale": 7.5,
                    "model": "stable-diffusion-v1-5"
                }
            }
            
        except Exception as e:
            print(f"Error generating image {index}: {str(e)}")
            return None

    def generate_dataset_images(self, descriptions_file: str, start_idx: int = 0, end_idx: int = None, 
                              batch_size: int = 1):
        """
        Generate images for all descriptions in the dataset
        
        Args:
            descriptions_file: Path to the JSON file containing image descriptions
            start_idx: Starting index for generation
            end_idx: Ending index for generation
            batch_size: Number of images to generate in parallel (use 1 for lower memory usage)
        """
        # Load descriptions
        with open(descriptions_file, 'r') as f:
            descriptions = json.load(f)
        
        if end_idx is None:
            end_idx = len(descriptions)
        
        # Initialize results tracking
        results = []
        
        # Generate images with progress bar
        for idx in tqdm(range(start_idx, end_idx), desc="Generating images"):
            desc = descriptions[idx]["description"]
            result = self.generate_single_image(desc, idx)
            
            if result:
                results.append({
                    **result,
                    "original_metadata": descriptions[idx]["metadata"]
                })
                
            # Save progress periodically
            if idx % 10 == 0:
                self._save_progress(results)
                
            # Small delay to prevent overheating
            time.sleep(0.5)
        
        # Save final results
        self._save_progress(results)
        
    def _save_progress(self, results: list):
        """Save generation progress to JSON file"""
        with open(os.path.join(self.output_dir, "generation_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

def main():
    # Initialize generator
    generator = LocalImageGenerator()
    
    # Generate images
    generator.generate_dataset_images(
        descriptions_file="retrieval_dataset/image_descriptions.json",
        start_idx=0,    # Start from beginning
        end_idx=10      # Generate first 10 images for testing
    )

if __name__ == "__main__":
    main()