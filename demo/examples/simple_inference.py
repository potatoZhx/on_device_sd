"""
Simple inference example demonstrating basic usage.
"""

from src.api.inference import MoEInferenceEngine

def main():
    # Initialize engine
    engine = MoEInferenceEngine(
        model_path="path/to/model",
        config_dir="configs"
    )
    
    # Generate text
    prompt = "Once upon a time"
    output_ids = engine.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8
    )
    
    print(f"Generated {len(output_ids)} tokens")
    print(f"Output IDs: {output_ids}")
    
    # Print statistics
    engine.print_statistics()


if __name__ == "__main__":
    main()