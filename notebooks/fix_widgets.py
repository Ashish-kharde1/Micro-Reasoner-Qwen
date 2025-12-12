import json

# Your specific file name
filename = 'qwen3_finetune.ipynb'

try:
    with open(filename, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Check if the 'widgets' key exists in metadata and remove it
    if 'metadata' in notebook and 'widgets' in notebook['metadata']:
        del notebook['metadata']['widgets']
        
        # Save the fixed version
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
            
        print(f"✅ Success! Removed widget metadata from {filename}.")
        print("You can now push this file to GitHub.")
    else:
        print(f"ℹ️  No widget metadata found in {filename}. It might already be clean.")

except FileNotFoundError:
    print(f"❌ Error: Could not find {filename} in this folder.")
except json.JSONDecodeError:
    print(f"❌ Error: The file {filename} is not valid JSON. It might be corrupted.")