import http.client
import json
import pandas as pd
import time
import re
import ssl
from tqdm import tqdm


# 1. Global Configuration
# =============================
API_KEY = "sk-VAatnjEd1hmbvfRzXoA0Uxonu59dK7OHK96JyZzuoOxkfKiw"
HOST = "globalai.vip"
MODEL = "gpt-4o-2024-05-13"


ssl_context = ssl._create_unverified_context()

# =============================
# 2. Extract probability function
# =============================
def extract_probability(text):
    try:
        match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0)?)\b", text)
        if match:
            return float(match.group(1))
    except:
        return None
    return None

# =============================
# 3. Load prompt
# =============================
df = pd.read_csv("data/llm_prompts.csv")
df["llm_response"] = ""
df["llm_probability"] = None

# =============================
# 4. main loop
# =============================
for i in tqdm(range(len(df))):
    prompt = df.loc[i, "prompt"]

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a traffic safety expert. Respond only with a probability between 0 and 1."},
            {"role": "user", "content": prompt}
        ]
    })

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Host": HOST,
        "Connection": "keep-alive"
    }

    try:
        conn = http.client.HTTPSConnection(HOST, context=ssl_context)
        conn.request("POST", "/v1/chat/completions", payload, headers)

        res = conn.getresponse()
        data = res.read().decode("utf-8")

        conn.close()

        reply_json = json.loads(data)
        reply = reply_json["choices"][0]["message"]["content"].strip()

        df.loc[i, "llm_response"] = reply
        df.loc[i, "llm_probability"] = extract_probability(reply)

        time.sleep(0.7)

    except Exception as e:
        print(f"Row {i} error: {e}")
        time.sleep(3)

    if i % 10 == 0:
        df.to_csv("llm_partial_gpt.csv", index=False)

# =============================
# 5. saved
# =============================
df.to_csv("data/llm_generated_gpt.csv", index=False)
print("Saved to llm_generated_gpt.csv")
