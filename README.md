# profile_analyzer

**映像と音声から人間性を分析し、マッチングアプリ向けプロフィール文を自動生成するAIツール**

---

## 概要

profile_analyzerは、動画ファイル（`.mp4`, `.mov`）と対応する音声（`.wav`）を分析するマルチモーダルAIツール

- Big Five性格分析（開放性、誠実性、外向性、協調性、神経症傾向）
- マルチモーダル誠実性分析（音声＋表情の一致）
- 統合スコアの算出
- マッチングアプリ用プロフィール文の自動生成

---

## プロジェクト構成

```
project_root/
├── target_videos/              # 分析対象の動画・音声を格納する
├── results/                    # 分析結果（JSON）および統合結果の出力先
├── profile_analyzer.py         # 個別分析用メインスクリプト
├── batch_analyze.py            # 一括分析・統合・プロフィール文生成用スクリプト
├── config.yaml                 # 設定ファイル
├── run.log                     # 実行ログ（自動生成）
├── requirements.txt            # 必要なPythonライブラリ
└── README.md                   # プロジェクト説明
```

---

## セットアップ手順

### 1. Python環境の準備

Python 3.10〜3.11を推奨します。

```bash
python --version
```

### 2. 仮想環境の構築

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 4. OpenAI APIキーの設定

プロジェクトのルートディレクトリに `.env` ファイルを作成し、以下の内容を記述してください。

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 実行方法

### 単一動画の分析

```bash
python profile_analyzer.py target_videos/your_video.mp4 target_videos/your_video.wav -o results/your_result.json
```

### 複数動画の一括分析とプロフィール生成

```bash
python batch_analyze.py
```

実行後、`results/integrated_result.json` に統合スコアとプロフィール文が出力されます。

---

## 出力例

```json
{
  "integrated_scores": {
    "openness": 4.2,
    "conscientiousness": 3.5,
    "extraversion": 4.8,
    "agreeableness": 4.0,
    "neuroticism": 2.0,
    "sincerity": 4.3
  },
  "matching_app_profile": "〇〇さんはとても社交的で自然体な人柄が魅力的。..."
}
```

---

## 設定ファイル (config.yaml)

WhisperやGPTモデル、フレームスキップなどの詳細設定が可能です。

```yaml
whisper_model: "base"
llm_model: "gpt-4o"
video_frame_skip: 5
ensure_ffmpeg: true

segmentation:
  max_chars: 120
  max_seconds: 15.0

weights:
  openness: 1.0
  conscientiousness: 1.0
  extraversion: 1.0
  agreeableness: 1.0
  neuroticism: 1.0
  sincerity: 1.0

sincerity_integration_weights:
  text: 0.5
  multimodal: 0.5

log_level: "INFO"
```

---

## 依存ライブラリ (requirements.txt)

```text
numpy==1.26.4
opencv-python==4.11.0.86
mediapipe==0.10.21
protobuf>=3.20.3,<5
openai-whisper==20231117
torch>=2.1,<2.4
tqdm>=4.65,<5
langchain==0.3.27
langchain-openai==0.3.30
pydantic>=2.3,<3
typing_extensions>=4.8.0
PyYAML>=6.0,<7
python-dotenv>=1.0,<2
tenacity>=8.2.3,<9
rich>=13.7,<14
```

---

## 主要技術スタック

- 音声認識: OpenAI Whisper  
- 表情解析: MediaPipe FaceMesh + OpenCV  
- 言語分析: LangChain × GPT-4  
- マルチモーダル評価: 独自セグメント分割＋モーション評価  
- プロフィール生成: GPT-4o によるプロンプト設計  

---

## 作者

- 名前: 飯田祥羽  
- 所属: 静岡大学 情報学部 竹内研究室  
- GitHub: [@kame447](https://github.com/kame447)

## 共同制作・サポート

- 名前:何美帆
- 所属: 静岡大学 情報学部 竹内研究室
- GitHub: [@Miho-ui](https://github.com/Miho-ui)
---

## ライセンス

MIT License  
本プロジェクトは自由に使用・改変・再配布が可能です。

---

## 今後の展望

- セグメント単位での信頼性可視化グラフの実装  
- 複数動画間における傾向・相関関係の可視化分析  
- 価値観や行動傾向に基づいたマッチングAIとの統合  
- ChatGPT等の対話型AIによる分析支援インタフェースの構築  
- クラスタリング手法による話者分離（人間とAIの音声を識別） 