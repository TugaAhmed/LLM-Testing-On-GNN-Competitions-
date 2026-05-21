# from openai import OpenAI
# import os

# client = OpenAI(
#     api_key=os.environ["NSCALE_API_KEY"],
#     base_url="https://inference.api.nscale.com/v1",
# )

# response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-Coder-32B-Instruct",
#     messages=[{"role": "user", "content": "Say hello."}],
#     max_tokens=10,
# )
# print(response.choices[0].message.content)

from pathlib import Path
SCRIPT_DIR = Path(__file__).parent                   
competition_DIR = SCRIPT_DIR.parent / "GNN_BACE_Challenge"  
csv_path = competition_DIR / "submission.csv"
csv_path2 = competition_DIR / "submissions" / "inbox" / "submission.csv"

print("csv_path",csv_path)
print("csv_path2",csv_path2)