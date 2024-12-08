import json
import random
import os
from typing import Dict, List

class RetrievalDatasetGenerator:
    def __init__(self):
        self.base_scenes = {
            "beach": {
                "subjects": ["children", "teenagers", "adults", "family", "couple", "group of friends"],
                "activities": {
                    "playing": [
                        "building elaborate sandcastles with moats",
                        "playing volleyball with colorful ball",
                        "playing frisbee near the shoreline",
                        "splashing in shallow waves",
                        "flying colorful kites in beach breeze",
                        "playing beach soccer",
                        "having water balloon fight",
                        "playing beach tennis",
                        "digging deep holes in sand",
                        "collecting seashells in buckets"
                    ],
                    "relaxing": [
                        "sunbathing on colorful towels",
                        "reading books under beach umbrella",
                        "napping in beach chairs",
                        "listening to music with headphones",
                        "meditating facing the ocean"
                    ],
                    "water_activities": [
                        "surfing on medium waves",
                        "bodyboarding in waves",
                        "swimming in clear water",
                        "snorkeling near coral reef",
                        "paddleboarding in calm water",
                        "kayaking along coastline"
                    ]
                },
                "time_of_day": ["early morning", "mid-morning", "noon", "afternoon", "sunset", "golden hour"],
                "weather": ["clear sunny", "partly cloudy", "overcast but bright", "misty morning"],
                "additional_elements": [
                    "colorful beach umbrellas nearby",
                    "seagulls flying overhead",
                    "palm trees swaying in breeze",
                    "lighthouse visible in distance",
                    "sailboats on horizon",
                    "rocky cliffs in background"
                ]
            },
            "park": {
                "subjects": ["children", "teenagers", "adults", "family", "elderly couple", "group of friends"],
                "activities": {
                    "playing": [
                        "playing soccer on grass field",
                        "throwing frisbee with dog",
                        "playing basketball on court",
                        "swinging on playground sets",
                        "playing hide and seek among trees",
                        "flying remote control planes",
                        "playing catch with baseball",
                        "doing gymnastics on grass",
                        "playing with bubble wands",
                        "rolling down grassy hills"
                    ],
                    "relaxing": [
                        "reading books on blanket",
                        "having picnic with basket",
                        "sketching in notepads",
                        "cloud watching while lying down",
                        "meditating under tree"
                    ],
                    "exercise": [
                        "jogging on walking trail",
                        "doing yoga on grass",
                        "practicing tai chi",
                        "doing bodyweight exercises",
                        "stretching near benches"
                    ]
                },
                "time_of_day": ["early morning", "mid-morning", "noon", "afternoon", "evening", "dusk"],
                "weather": ["sunny spring day", "autumn with falling leaves", "mild summer evening", "clear winter afternoon"],
                "additional_elements": [
                    "flowering cherry trees",
                    "ducks swimming in pond",
                    "squirrels gathering nuts",
                    "colorful flower gardens",
                    "stone pathway winding through",
                    "wooden gazebo nearby"
                ]
            }
        }

    def generate_descriptions(self, num_descriptions: int = 1000) -> List[Dict]:
        descriptions = []
        
        while len(descriptions) < num_descriptions:
            scene = random.choice(list(self.base_scenes.keys()))
            scene_data = self.base_scenes[scene]
            
            subject = random.choice(scene_data["subjects"])
            activity_type = random.choice(list(scene_data["activities"].keys()))
            specific_activity = random.choice(scene_data["activities"][activity_type])
            time = random.choice(scene_data["time_of_day"])
            weather = random.choice(scene_data["weather"])
            additional = random.choice(scene_data["additional_elements"])
            
            desc = f"{subject} {specific_activity} during {time}, {weather}, with {additional}"
            
            description_entry = {
                "description": desc,
                "metadata": {
                    "scene": scene,
                    "subject": subject,
                    "activity_type": activity_type,
                    "specific_activity": specific_activity,
                    "time": time,
                    "weather": weather,
                    "additional_element": additional
                }
            }
            
            if description_entry not in descriptions:
                descriptions.append(description_entry)
        
        return descriptions

    def generate_evaluation_queries(self) -> List[Dict]:
        queries = []
        
        # Specific activity queries
        for scene in self.base_scenes.keys():
            for activity_type in self.base_scenes[scene]["activities"].keys():
                for specific_activity in random.sample(self.base_scenes[scene]["activities"][activity_type], 2):
                    base_activity = specific_activity.split()[0]  # Get first word of activity
                    queries.append({
                        "query": f"Show me {base_activity} at {scene}",
                        "type": "general_activity",
                        "target": {"scene": scene, "activity_type": activity_type}
                    })
                    queries.append({
                        "query": specific_activity,
                        "type": "specific_activity",
                        "target": {"scene": scene, "specific_activity": specific_activity}
                    })

        # Subject-focused queries
        for scene in self.base_scenes.keys():
            for subject in random.sample(self.base_scenes[scene]["subjects"], 2):
                queries.append({
                    "query": f"{subject} at {scene}",
                    "type": "subject_focus",
                    "target": {"scene": scene, "subject": subject}
                })

        # Time and weather queries
        for scene in self.base_scenes.keys():
            time = random.choice(self.base_scenes[scene]["time_of_day"])
            weather = random.choice(self.base_scenes[scene]["weather"])
            queries.append({
                "query": f"{scene} during {time}",
                "type": "time_focus",
                "target": {"scene": scene, "time": time}
            })
            queries.append({
                "query": f"{scene} on {weather}",
                "type": "weather_focus",
                "target": {"scene": scene, "weather": weather}
            })

        # Complex queries
        complex_templates = [
            "{subject} {specific_activity} at {scene}",
            "{subject} {activity_type} during {time}",
            "{specific_activity} on {weather}",
            "{activity_type} at {scene} during {time}"
        ]

        for template in complex_templates:
            scene = random.choice(list(self.base_scenes.keys()))
            scene_data = self.base_scenes[scene]
            activity_type = random.choice(list(scene_data["activities"].keys()))
            specific_activity = random.choice(scene_data["activities"][activity_type])
            
            query_text = template.format(
                subject=random.choice(scene_data["subjects"]),
                specific_activity=specific_activity,
                activity_type=activity_type,
                scene=scene,
                time=random.choice(scene_data["time_of_day"]),
                weather=random.choice(scene_data["weather"])
            )
            
            queries.append({
                "query": query_text,
                "type": "complex",
                "target": {"scene": scene, "activity_type": activity_type}
            })

        return queries[:100]

def main():
    output_dir = "retrieval_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    generator = RetrievalDatasetGenerator()
    descriptions = generator.generate_descriptions(1000)
    queries = generator.generate_evaluation_queries()
    
    with open(f"{output_dir}/image_descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=2)
    
    with open(f"{output_dir}/evaluation_queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    
    print(f"Generated {len(descriptions)} image descriptions")
    print(f"Generated {len(queries)} evaluation queries")
    print(f"Files saved in '{output_dir}' directory")
    
    print("\nSample Image Descriptions:")
    for desc in random.sample(descriptions, 3):
        print(f"- {desc['description']}")
    
    print("\nSample Evaluation Queries:")
    for query in random.sample(queries, 3):
        print(f"- {query['query']}")

if __name__ == "__main__":
    main()