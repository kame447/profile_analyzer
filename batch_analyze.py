import os
import sys
import subprocess
import json
import glob
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIキーのチェック
if os.getenv("OPENAI_API_KEY") is None:
    print("エラー: OPENAI_API_KEYが環境変数に設定されていません。")
    exit()

# 定数定義
TARGET_DIR = "./target_videos"
RESULTS_DIR = "./results"
PROFILE_ANALYZER_SCRIPT = "profile_analyzer.py"
OUTPUT_JSON_FILE = "integrated_result.json"

# LangChainの初期化
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# プロンプトテンプレート
PROMPT_TEMPLATE = """
あなたは、複数の動画分析結果と統合された性格スコアを基に、マッチングアプリのプロフィール文を作成するプロのコピーライターです。
以下の分析結果から、この人物の魅力を最大限に引き出す、親しみやすく、興味を引くような紹介文を作成してください。さらに、分析結果から推測される「相性の良い人物像」についても言及してください。

### 分析結果
{analysis_results}

### 統合スコア
{integrated_scores}

### マッチングアプリ向けプロフィール文
フォーマット：
- 冒頭：この人物の最も魅力的な一面を簡潔に表現するキャッチーな一文。
- 本文：性格の主要な特徴を、具体的な行動や趣味、価値観と結びつけて記述する。
- 相性の良い人物像：分析結果から推測される、どんな性格や価値観を持つ人物と相性が良いかについて言及する。
- 最後に、全体をまとめる簡潔な一文。
- 全体的に、話し言葉のような親しみやすいトーンで、読み手が「この人と話してみたい！」と思うような内容にしてください。
"""

# ファイル一覧を取得
def find_files():
    return glob.glob(os.path.join(TARGET_DIR, "*.mov")) + glob.glob(os.path.join(TARGET_DIR, "*.mp4"))

# .mov から mp4/wav に変換
def convert_mov_to_mp4_wav(mov_file):
    base_name = os.path.splitext(os.path.basename(mov_file))[0]
    mp4_file = os.path.join(TARGET_DIR, f"{base_name}.mp4")
    wav_file = os.path.join(TARGET_DIR, f"{base_name}.wav")

    if os.path.exists(mp4_file) and os.path.exists(wav_file):
        print(f"変換済み: {mov_file}")
        return mp4_file, wav_file

    print(f"{mov_file} を変換中...")
    try:
        subprocess.run(["ffmpeg", "-i", mov_file, "-c:v", "copy", "-an", mp4_file], check=True)
        subprocess.run(["ffmpeg", "-i", mov_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", wav_file], check=True)
        print(f"変換成功: {mp4_file}, {wav_file}")
        return mp4_file, wav_file
    except subprocess.CalledProcessError as e:
        print(f"変換失敗: {mov_file}, エラー: {e}")
        return None, None

# 分析実行
def run_profile_analyzer(mp4_file, wav_file, result_path):
    print(f"{os.path.basename(mp4_file)} の個別分析を実行中...")
    try:
        command = [sys.executable, PROFILE_ANALYZER_SCRIPT, mp4_file, wav_file, "-o", result_path]
        subprocess.run(command, check=True)
        print(f"個別分析成功: {os.path.basename(mp4_file)}")
    except subprocess.CalledProcessError as e:
        print(f"個別分析失敗: {os.path.basename(mp4_file)}, エラー: {e}")

# JSONファイルを集める
def get_all_result_jsons():
    return glob.glob(os.path.join(RESULTS_DIR, "*_result.json"))

# 分析結果の統合
def integrate_results(json_files):
    all_results = []
    invalid_files = [] # 無効なファイルを格納するリスト
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data["__source_file__"] = os.path.basename(json_file)
                all_results.append(data)
        except json.JSONDecodeError as e:
            print(f"エラー: 無効なJSONファイルが見つかりました -> {json_file}")
            print(f"詳細: {e}")
            invalid_files.append(json_file)
            continue
        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません -> {json_file}")
            invalid_files.append(json_file)
            continue
    
    if not all_results:
        return None, None, invalid_files

    integrated_scores = {}
    counts = {}
    for result in all_results:
        for key, value in result.get("overall_evaluations", {}).items():
            if isinstance(value, dict) and 'score' in value and isinstance(value['score'], (int, float)):
                integrated_scores[key] = integrated_scores.get(key, 0) + value['score']
                counts[key] = counts.get(key, 0) + 1

    for key in integrated_scores:
        if counts[key] > 0:
            integrated_scores[key] /= counts[key]
        else:
            integrated_scores[key] = 0

    analysis_results_text = ""
    for i, result in enumerate(all_results):
        file_id = result.get("__source_file__", f"result_{i}").replace("_result.json", "")
        analysis_results_text += f"--- 分析結果 {i+1}: {file_id} ---\n"
        analysis_results_text += json.dumps({
            "overall_evaluations": result.get("overall_evaluations", {})
        }, indent=2, ensure_ascii=False) + "\n\n"

    return analysis_results_text, integrated_scores, invalid_files

# プロフィール生成
def generate_introduction(analysis_results, integrated_scores):
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(PROMPT_TEMPLATE))
    print("GPTでプロフィール文を生成中...")
    response = chain.run(
        analysis_results=analysis_results,
        integrated_scores=json.dumps(integrated_scores, indent=2, ensure_ascii=False)
    )
    match = re.search(r"### マッチングアプリ向けプロフィール文\s+(.*)", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()

# メイン処理
def main():
    print("--- batch_analyze.py を開始します ---")
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_files = get_all_result_jsons()
    
    # 既存のJSONファイルが3つ以上あれば、個別分析をスキップ
    if len(json_files) >= 3:
        print("results/ フォルダに3つ以上の分析結果が存在するため、個別分析をスキップし、統合処理を開始します。")
    else:
        video_files = find_files()
        processed_set = set()
        processed_files = []

        for file in video_files:
            base_name = os.path.splitext(os.path.basename(file))[0]
            if base_name in processed_set:
                continue

            if file.endswith(".mov"):
                mp4, wav = convert_mov_to_mp4_wav(file)
                if mp4 and wav:
                    processed_files.append((base_name, mp4, wav))
                    processed_set.add(base_name)
            elif file.endswith(".mp4"):
                wav = os.path.join(TARGET_DIR, f"{base_name}.wav")
                if os.path.exists(wav):
                    processed_files.append((base_name, file, wav))
                    processed_set.add(base_name)
                else:
                    print(f"エラー: {file} に対応する.wavが見つかりません。スキップします。")

        for base_name, mp4, wav in processed_files:
            output_path = os.path.join(RESULTS_DIR, f"{base_name}_result.json")
            if not os.path.exists(output_path):
                run_profile_analyzer(mp4, wav, output_path)
            else:
                print(f"情報: 既に分析済みです -> {output_path}")

    json_files = get_all_result_jsons()
    analysis_text, scores, invalid_files = integrate_results(json_files)

    if analysis_text and scores:
        intro = generate_introduction(analysis_text, scores)
        output = {
            "integrated_scores": scores,
            "matching_app_profile": intro
        }
        with open(os.path.join(RESULTS_DIR, OUTPUT_JSON_FILE), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"成功: 統合結果を保存しました -> {os.path.join(RESULTS_DIR, OUTPUT_JSON_FILE)}")
    else:
        print("エラー: 統合処理失敗しました。")
        print("原因: 有効な分析結果が1つも見つからなかったため。")
        if invalid_files:
            print("以下のファイルが原因で統合に失敗しました:")
            for f in invalid_files:
                print(f"- {f}")
        else:
            print("results/フォルダ内に有効なJSONファイルが存在しません。")

    print("--- 処理完了 ---")

if __name__ == "__main__":
    main()