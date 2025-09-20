import json
import glob

def clean_text(text: str) -> str:
    # Remove leading/trailing spaces and normaize spaes/newlines
    return " ".join(text.split())

def convert_to_jsonl(input_files, output_file):
    with open(output_file, 'w', encoding = 'utf-8') as out_f:
        for file in input_files:
            print (f"Processing {file}")
            with open(file, 'r', encoding = 'utf-8') as f:
                data = json.load(f)

            for obj in data:
                if "text" in obj:
                    obj["text"] = clean_text(obj["text"])

                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Merged {len(input_files)} files into {output_file}")


if __name__ == "__main__":
    input_files = glob.glob("./datasets/language_detection/*.json") # Picks all .json files in the folder
    convert_to_jsonl(input_files, "./datasets/language_detection/merged_dataset.jsonl")
