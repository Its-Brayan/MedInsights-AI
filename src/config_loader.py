import yaml
def load_config(filename="config.yaml"):
    path = f"/home/brayan/Aiprojects/MedInsights-AI/config/{filename}"

    with open(path,"r") as file:
        config = yaml.safe_load(file)

    return config
def load_prompt(filename="prompt_config.yaml"):
    path=f"/home/brayan/Aiprojects/MedInsights-AI/config/{filename}"
    with open(path,"r") as file:
        prompt_config = yaml.safe_load(file)
    
    return prompt_config