
import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def split_kn(kn_text: str):
    """
    Split <EOS>-separated KN text into individual sentences.
    Remove extra whitespace and empty entries.
    """
    if not isinstance(kn_text, str):
        return []
    return [s.strip() for s in kn_text.split("<EOS>") if s.strip()]

def query_gpt4o_rank(hs_text: str, kn_list: list[str], model: str = "gpt-4o-mini") -> dict:
    """
    Ask GPT-4o to rank each knowledge sentence by relevance to HS.
    Returns a dictionary {index: score}.
    """
    prompt = f"""
You are a multilingual assistant selecting background knowledge
sentences that best support generating a respectful, factual, and empathetic counter-narrative.

Rate each Knowledge Sentence (KN) from 0 (irrelevant) to 5 (highly relevant)
based on how useful it is to respond accurately and compassionately to the Hate Speech (HS).

Return ONLY a JSON list like this:
[
  {{"index": 1, "score": 5, "reason": "Directly refutes the hate claim."}},
  {{"index": 2, "score": 3, "reason": "Somewhat related."}},
  ...
]

HATE SPEECH:
{hs_text}

KNOWLEDGE SENTENCES:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(kn_list)])}
    """.strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw_output = response.choices[0].message.content.strip()

        # Parse JSON safely
        try:
            data = json.loads(raw_output)
            scores = {d["index"] - 1: int(d.get("score", 0)) for d in data if "index" in d}
        except Exception:
            print(" Parse error, using empty scores. Raw output:")
            print(raw_output)
            scores = {}

        return scores

    except Exception as e:
        print(" GPT-4o API error:", e)
        return {}

def select_top_n(sentences: list[str], scores: dict, n: int = 2) -> list[str]:
    """Return top-n sentences based on GPT scores."""
    if not sentences:
        return []
    if not scores:
        return sentences[:n]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_idxs = [idx for idx, _ in ranked[:n]]
    return [sentences[i] for i in top_idxs if i < len(sentences)]

def process_split(csv_path: str, out_path: str, model: str = "gpt-4o-mini"):
    """Run GPT-4o ranking for each example in the split."""
    df = pd.read_csv(csv_path)
    print(f"📄 Processing {csv_path} ({len(df)} rows)")

    selected_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        hs = row["HS"]
        kn_sentences = split_kn(row["KN"])
        if not kn_sentences:
            row["KN_SELECTED"] = ""
            selected_data.append(row)
            continue

        scores = query_gpt4o_rank(hs, kn_sentences, model)
        selected_kn = select_top_n(kn_sentences, scores, n=2)
        row["KN_SELECTED"] = " <EOS> ".join(selected_kn)
        selected_data.append(row)

    out_df = pd.DataFrame(selected_data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    splits = ["train", "validation", "test"]
    for split in splits:
        in_path = f"data/raw/{split}.csv"
        out_path = f"data/processed/{split}_selected.csv"
        if os.path.exists(in_path):
            process_split(in_path, out_path)
        else:
            print(f"Missing {in_path}, skipping.")
