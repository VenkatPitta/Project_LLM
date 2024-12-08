import os
import json
import torch
import numpy as np
import faiss
import sqlite3
from PIL import Image
from datetime import datetime
import open_clip

class EmbeddingGenerator:
    def __init__(self, image_dir: str, cache_dir: str = "./embeddings_cache"):
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.model_dir = os.path.join(cache_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize CLIP model and preprocessing
        self.model_name = "ViT-B-32"
        self.pretrained = "laion2b_s34b_b79k"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.pretrained}.pt")
        
        # Create model, preprocess, and tokenizer
        self.model, self.preprocess, self.tokenizer = self.load_or_download_model()
        
        # Cache file paths
        self.db_path = os.path.join(cache_dir, "embeddings.db")
        self.index_file = os.path.join(cache_dir, "faiss_index.bin")
        
        # Initialize FAISS index
        self.embedding_dim = 512  # CLIP's embedding dimension
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Setup database and load index
        self.setup_database()
        self.load_faiss_index()
    
    def load_or_download_model(self):
        """Load model from local storage or download if not available"""
        try:
            # Create model, preprocess, and tokenizer
            model, preprocess, tokenizer = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            
            if os.path.exists(self.model_path):
                print(f"Loading model from local storage: {self.model_path}")
                # Load local weights with weights_only=True for security
                model.load_state_dict(torch.load(self.model_path, weights_only=True))
                print("Model loaded successfully from local storage")
            else:
                print("Downloading model for the first time...")
                # Save model weights locally
                torch.save(model.state_dict(), self.model_path)
                print(f"Model saved to: {self.model_path}")
            
            return model, preprocess, tokenizer
            
        except Exception as e:
            print(f"Error loading/downloading model: {str(e)}")
            raise

    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                    (image_name TEXT PRIMARY KEY,
                     embedding BLOB,
                     faiss_id INTEGER,
                     created_at TIMESTAMP)''')
        conn.commit()
        conn.close()
    
    def load_faiss_index(self):
        """Load existing FAISS index if available"""
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
            except Exception as e:
                print(f"Error loading FAISS index: {str(e)}")
                self.index = faiss.IndexFlatIP(self.embedding_dim)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image for CLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error preprocessing {image_path}: {str(e)}")
            return None

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image"""
        try:
            preprocessed_image = self.preprocess_image(image_path)
            if preprocessed_image is None:
                return None
            
            with torch.no_grad():
                embedding = self.model.encode_image(preprocessed_image)
                embedding = embedding.cpu().numpy()
                faiss.normalize_L2(embedding)
                return embedding
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {str(e)}")
            return None

    def get_processed_images(self) -> set:
        """Get list of already processed images from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT image_name FROM embeddings')
        processed = {row[0] for row in c.fetchall()}
        conn.close()
        return processed

    def update_embeddings(self):
        """Process only new images and update database and FAISS index"""
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        processed_images = self.get_processed_images()
        new_files = [f for f in image_files if f not in processed_images]
        
        if not new_files:
            print("No new images to process.")
            return
        
        print(f"Processing {len(new_files)} new images...")
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT MAX(faiss_id) FROM embeddings')
        last_id = c.fetchone()[0]
        current_id = (last_id if last_id is not None else -1) + 1
        
        new_embeddings = []
        
        for img_file in new_files:
            img_path = os.path.join(self.image_dir, img_file)
            embedding = self.generate_embedding(img_path)
            
            if embedding is not None:
                new_embeddings.append(embedding)
                c.execute('''INSERT INTO embeddings 
                            (image_name, embedding, faiss_id, created_at)
                            VALUES (?, ?, ?, ?)''',
                         (img_file, 
                          embedding.tobytes(), 
                          current_id,
                          datetime.now().isoformat()))
                current_id += 1
                print(f"Generated embedding for {img_file}")
        
        if new_embeddings:
            new_embeddings = np.vstack(new_embeddings)
            self.index.add(new_embeddings)
            faiss.write_index(self.index, self.index_file)
            conn.commit()
            print("Database and FAISS index updated successfully")
        
        conn.close()