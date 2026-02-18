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

BATCH_SIZE = 50  # 今回の修正対象行数（APIのトークン制限を考慮して調整）
OVERLAP_SIZE = 5 # 前のバッチから持ち越す「参考」行数
SLEEP_TIME = 5   # API呼び出し間の待機時間（秒）無料版レート制限対策

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

def process_batch_with_llm(target_texts, context_texts, rules_dict):
    """
    ID飛び（結合）対策強化版
    """
    # 1. 入力をID付きオブジェクトに変換
    structured_target = [{"id": i, "text": t} for i, t in enumerate(target_texts)]
    
    rules_str = "\n".join([f"- 「{k}」→「{v}」" for k, v in rules_dict.items()])
    
    prompt = f"""
あなたは強力なテキスト校正エンジンです。
入力された JSONデータの `text` フィールドを修正し、同じ JSON構造で出力してください。

【最重要ルール：行の独立性維持】
* **いかなる理由があっても、行（ID）を結合・削除してはいけません。**
* 文の途中で改行されている場合、**文法的に不自然であっても、そのままの改行位置を維持してください。**
* 「前の行とつなげたほうが読みやすい」という判断は**禁止**です。

【textフィールドの修正指示】
1. **フィラー削除:** 「えー」「あのー」などの無意味な言葉は削除してください。
2. **ルール適用:** 表記ルールを適用してください。
3. **自然な日本語:** 文意を変えない範囲で整えてください。ただし、改行位置の移動は禁止です。

【表記ルール】
{rules_str}

【参考（直前の流れ）】
{json.dumps(context_texts, ensure_ascii=False)}

【修正対象データ】
{json.dumps(structured_target, ensure_ascii=False)}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            config={'response_mime_type': 'application/json'},
            contents=prompt
        )
        
        results_json = json.loads(response.text)
        
        # IDをキーにした辞書に変換（検索を高速化）
        result_map = {item.get('id'): item.get('text', '') for item in results_json}
        
        fixed_texts = []
        for i in range(len(target_texts)):
            if i in result_map:
                # 正常に修正されたテキスト
                fixed_texts.append(result_map[i])
            else:
                # 【重要】IDが飛んでいる（結合された）場合
                # 無理に分割しようとせず、オリジナルのテキストをそのまま採用する
                # これにより、字幕のタイミングズレ（ドミノ倒し）を防ぐ
                print(f"警告: ID {i} が消失しました。同期ズレを防ぐため原文を使用します: '{target_texts[i]}'")
                fixed_texts.append(target_texts[i])
                
        return fixed_texts

    except Exception as e:
        print(f"APIエラー: {e}")
        # エラー時は全行オリジナルを返す（安全策）
        return target_texts

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
    subs = pysrt.open(str(input_path), encoding='utf-8')
    total = len(subs)

    previous_fixed_lines = [] # 前のバッチから持ち越す「参考」行を保存するリスト

    print(f"モデル: {MODEL_ID}")
    print(f"対応表 {RULES_JSON} から {len(rules)} 件のルールを適用します。")

    print(f"処理を開始します: {input_path} -> {output_path}")
    print(f"{total}行 / バッチ:{BATCH_SIZE} / 重複:{OVERLAP_SIZE}")

    for i in range(0, total, BATCH_SIZE):
        batch = subs[i:i+BATCH_SIZE]
        target_texts = [s.text for s in batch]
        
        print(f"処理中: {i+1}〜{min(i+BATCH_SIZE, total)} / {total} 行")
        
        fixed_texts = process_batch_with_llm(target_texts, previous_fixed_lines, rules)
        
        for j, s in enumerate(batch):
            if j < len(fixed_texts):
                s.text = fixed_texts[j]
        
        previous_fixed_lines = fixed_texts[-OVERLAP_SIZE:]

        # 無料版レート制限(15RPM)対策
        time.sleep(SLEEP_TIME)

    subs.save(str(output_path), encoding='utf-8')
    print(f"完了しました: {output_path}")

if __name__ == "__main__":
    main()
