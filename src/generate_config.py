from transformers import GPT2Config
import json

config = GPT2Config()  # Use the correct config class
config_dict = config.to_dict()

# Save config.json
with open("./save_model/config.json", "w") as f:
    json.dump(config_dict, f, indent=4)

print("config.json created successfully!")
