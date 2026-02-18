import os
import json
import time
import argparse
from pathlib import Path
import pysrt
from google import genai
from dotenv import load_dotenv

# --- 初期設定 ---
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
RULES_JSON = "rules.json"
# 無料版制限に合わせた最適設定
BATCH_SIZE = 60 
SLEEP_TIME = 5

if not API_KEY:
    raise ValueError("APIキーが見つかりません。.envファイルに GEMINI_API_KEY を設定してください。")

# 新しいSDKのクライアント初期化
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.5-flash"

def load_rules(file_path):
    """JSON形式の対応表を読み込む"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def process_batch_with_llm(texts, rules_dict):
    """JSONモードを使用して、確実に行数と内容を一致させる"""
    rules_str = "\n".join([f"- 「{k}」→「{v}」" for k, v in rules_dict.items()])
    
    prompt = f"""
あなたは動画字幕の専門編集者です。
入力された {len(texts)} 行の字幕を、以下の指示に従って修正し、必ず【JSONの文字列配列】として出力してください。

【指示】
1. 「えー」「あのー」などのフィラーを削除する。
2. 表記ルールを適用する。
3. 自然な日本語に整える。
4. 「1:」などの番号や解説、太字装飾は一切含めないこと。
5. 入力と同じ {len(texts)} 要素の配列を維持すること。

【表記ルール】
{rules_str}

【入力テキスト】
{json.dumps(texts, ensure_ascii=False)}
"""

    try:
        # response_mime_type を指定して、必ずJSONで返させる
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
            }
        )
        
        # 文字列として返ってきたJSONをPythonのリストに変換
        results = json.loads(response.text)
        
        # 万が一、行数が一致しない場合のログ
        if len(results) != len(texts):
            print(f"警告: 行数が一致しません (入力:{len(texts)}, 出力:{len(results)})")
            
        return results
    except Exception as e:
        print(f"APIエラーまたはJSON解析エラー: {e}")
        return texts
    
def main():
    parser = argparse.ArgumentParser(description="SRTファイルをLLM(google-genai)で修正します。")
    parser.add_argument("input_file", help="入力するSRTファイル名")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"エラー: ファイル {input_path} が見つかりません。")
        return

    # 出力ファイル名の自動生成 (video.srt -> video_out.srt)
    output_path = input_path.with_name(f"{input_path.stem}_out{input_path.suffix}")

    rules = load_rules(RULES_JSON)
    print(f"モデル: {MODEL_ID}")
    print(f"対応表 {RULES_JSON} から {len(rules)} 件のルールを適用します。")

    subs = pysrt.open(str(input_path), encoding='utf-8')
    total = len(subs)
    
    print(f"処理を開始します: {input_path} -> {output_path}")

    for i in range(0, total, BATCH_SIZE):
        batch = subs[i:i+BATCH_SIZE]
        texts = [s.text for s in batch]
        
        print(f"処理中: {i+1}〜{min(i+BATCH_SIZE, total)} / {total} 行")
        
        fixed_texts = process_batch_with_llm(texts, rules)
        
        for j, s in enumerate(batch):
            if j < len(fixed_texts):
                s.text = fixed_texts[j]
        
        # 無料版レート制限(15RPM)対策
        time.sleep(SLEEP_TIME)

    subs.save(str(output_path), encoding='utf-8')
    print(f"完了しました！")

if __name__ == "__main__":
    main()
