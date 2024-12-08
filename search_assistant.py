import os
import torch
import numpy as np
from PIL import Image
import anthropic
from typing import List, Tuple, Dict
import faiss
import sqlite3
from dotenv import load_dotenv
import open_clip

class SearchAssistant:
    def __init__(self, embedding_generator, db_path: str, top_k: int = 5):
        load_dotenv()
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        self.embedding_generator = embedding_generator
        self.db_path = db_path
        self.top_k = top_k
        
        self.system_prompt = """You are helping refine image search queries for CLIP, a visual-language model. 
        Create a clear, descriptive query that combines all relevant information from the conversation history.
        Focus on visual elements that would be present in images. Be specific but concise.
        Use only information explicitly mentioned in the conversation - do not add assumptions or extra details.
        Format: Return only the refined query text, no other words or explanation."""
        
        self.conversation_history = []
    
    def get_refined_query(self, user_input: str, feedback: str = None) -> str:
        """Get CLIP-friendly description based on conversation history"""
        # Add latest interaction to history
        self.conversation_history.append(user_input)
        if feedback:
            self.conversation_history.append(feedback)
            
        # Combine conversation history into context
        conversation_context = " | Previous search context: ".join(self.conversation_history)
        
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            temperature=0.0,
            system=self.system_prompt,
            messages=[{"role": "user", "content": f"Create a refined image search query using this conversation history: {conversation_context}"}]
        )
        
        return response.content[0].text.strip()
    
    def get_image_embeddings(self, text_description: str) -> np.ndarray:
        """Generate CLIP embeddings for the text description"""
        with torch.no_grad():
            text = open_clip.tokenize([text_description]).to(self.embedding_generator.device)
            text_features = self.embedding_generator.model.encode_text(text)
            text_features = text_features.cpu().numpy()
            faiss.normalize_L2(text_features)
        return text_features
    
    def search_images(self, text_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Search for similar images using FAISS index"""
        distances, indices = self.embedding_generator.index.search(text_embedding, self.top_k)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            c.execute('SELECT image_name FROM embeddings WHERE faiss_id = ?', (int(idx),))
            result = c.fetchone()
            if result:
                image_name = result[0]
                similarity = float(distance)
                results.append((image_name, similarity))
        
        conn.close()
        return results
    
    def interactive_search(self):
        """Run interactive search session with user"""
        print("Welcome! What kind of photos are you looking for?")
        
        # Reset conversation history at start of new search
        self.conversation_history = []
        
        user_query = input("You: ")
        feedback = None
        iteration = 0
        max_iterations = 5
        
        while iteration < max_iterations:
            # Get refined query from Claude
            refined_query = self.get_refined_query(user_query, feedback)
            print(f"\nSearching for: {refined_query}")
            
            # Generate embeddings and search
            text_embedding = self.get_image_embeddings(refined_query)
            results = self.search_images(text_embedding)
            
            # Show results
            print("\nFound these matches:")
            for i, (image_name, similarity) in enumerate(results, 1):
                print(f"{i}. {image_name} (similarity: {similarity:.3f})")
            
            # Get feedback
            print("\nAre these results what you're looking for?")
            print("1. Yes, these are perfect")
            print("2. No, there is something different")
            print("3. Exit search")
            
            choice = input("\nYour choice (1-3): ")
            
            if choice == "1":
                print("Great! Glad I could help!")
                break
            elif choice == "2":
                feedback = input("\nPlease tell me what's should be there: ")
                iteration += 1
            elif choice == "3":
                print("Thank you for using the search assistant!")
                break
            else:
                print("Invalid choice. Please try again.")
            
            if iteration == max_iterations:
                print("\nReached maximum number of refinements. Please try a new search.")
                break
        
        return results

if __name__ == "__main__":
    from generate_embeddings import EmbeddingGenerator
    
    # Initialize embedding generator
    image_dir = "./photos"  # Change this to your photos directory
    generator = EmbeddingGenerator(image_dir)
    
    # Update embeddings for any new images
    generator.update_embeddings()
    
    # Initialize search assistant
    assistant = SearchAssistant(generator, generator.db_path)
    
    # Start interactive search
    assistant.interactive_search()