import os

for filename in os.listdir("logs"):
    filepath = os.path.join("logs", filename)
    name, extension = os.path.splitext(filepath)
    if filename.endswith(".dot"):
        os.system(f"dot -Tsvg -o {name}.svg {name}.dot")
