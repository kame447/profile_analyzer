import os
import sys
import json
import argparse
import time
import logging
import shutil
import inspect
import uuid
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Union

import cv2
import yaml
import numpy as np
import mediapipe as mp
import whisper
import torch
import langchain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from rich.progress import Progress

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY が見つかりません。.env または環境変数で設定してください。")

# 設定スキーマとヘルパー関数

class AppConfig(BaseModel):
    """アプリケーション設定のPydanticスキーマ"""
    whisper_model: str = "base"
    llm_model: str = "gpt-4o"
    video_frame_skip: int = 5
    ensure_ffmpeg: bool = True
    segmentation: Dict[str, Union[int, float]]
    weights: Dict[str, float]
    sincerity_integration_weights: Dict[str, float]
    log_level: str = "INFO"

def load_config(path: str = "config.yaml") -> AppConfig:
    """config.yamlを読み込み、Pydanticスキーマで検証する"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
            return AppConfig(**config_data)
    except FileNotFoundError:
        logging.warning(f"設定ファイル {path} が見つかりません。デフォルト設定を使用します。")
        return AppConfig(segmentation={"max_chars": 120, "max_seconds": 15.0},
                         weights={"openness": 1.0, "conscientiousness": 1.0, "extraversion": 1.0, "agreeableness": 1.0, "neuroticism": 1.0, "sincerity_text": 1.0, "sincerity_multimodal": 1.0},
                         sincerity_integration_weights={"text": 0.5, "multimodal": 0.5})
    except ValidationError as e:
        logging.critical(f"設定ファイルの検証に失敗しました。構造または値を確認してください:\n{e.errors()}")
        raise
    except Exception as e:
        logging.critical(f"設定ファイルのロード中に予期せぬエラーが発生しました: {e}")
        raise

def to_dict(model_instance: BaseModel) -> Dict[str, Any]:
    return model_instance.model_dump() if hasattr(model_instance, "model_dump") else model_instance.dict()

def safe_float(val: Any) -> Optional[float]:
    """NumPy型を安全にfloatに変換するユーティリティ"""
    try:
        if val is None or (isinstance(val, (int, float, np.number)) and np.isnan(val)):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None

def _validate_inputs(video_p: Path, audio_p: Path, max_mb=2048):
    video_ext_ok = video_p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
    audio_ext_ok = audio_p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
    for p, name, ok in [(video_p, "video", video_ext_ok), (audio_p, "audio", audio_ext_ok)]:
        if not p.exists(): raise FileNotFoundError(f"{name} が見つかりません: {p}")
        if not ok: logging.warning(f"{name} の拡張子が想定外です: {p.suffix}")
        if p.stat().st_size > max_mb * 1024**2: logging.warning(f"{name} ファイルサイズが大きいです ({p.stat().st_size / 1024**2:.1f}MB)")

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    w = {k: max(0.0, float(v)) for k, v in w.items()}
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}

# config.yaml の設定を読み込み
try:
    config = load_config()
except Exception:
    sys.exit(1)

WHISPER_MODEL_NAME = config.whisper_model
LLM_MODEL_NAME = config.llm_model
WEIGHTS = _normalize_weights(config.weights)
SINCERITY_INTEGRATION_WEIGHTS = config.sincerity_integration_weights
VIDEO_FRAME_SKIP = config.video_frame_skip
log_level = config.log_level.upper()
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
file_handler = logging.FileHandler("run.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
root_logger.addHandler(file_handler)

# ツール（映像・音声処理） 
def analyze_face_landmarks(video_path: str) -> Dict[str, Any]:
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"動画ファイルが開けませんでした: {video_path}")
        return {"features": {}, "note": "Video file could not be opened."}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps == 0.0: logging.warning("FPSが0のため、durationを計算できません。")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    all_landmarks, processed_frames_indices = [], []

    face_mesh_kwargs = dict(static_image_mode=False, max_num_faces=1,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if "refine_landmarks" in inspect.signature(mp.solutions.face_mesh.FaceMesh).parameters:
        face_mesh_kwargs["refine_landmarks"] = True

    with mp.solutions.face_mesh.FaceMesh(**face_mesh_kwargs) as face_mesh:
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            if frame_count % VIDEO_FRAME_SKIP == 0:
                processed_frames_indices.append(frame_count)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]
                    all_landmarks.append(landmarks)
            frame_count += 1
    cap.release()

    if not all_landmarks:
        logging.warning("動画から顔が検出されませんでした。")
        return {"features": {}, "note": "No faces detected."}
        
    landmarks_np = np.array(all_landmarks, dtype=np.float32)
    mean_coords_per_frame = np.mean(landmarks_np, axis=1)
    motions = np.linalg.norm(np.diff(mean_coords_per_frame, axis=0), axis=1)
    frame_timestamps = (np.array(processed_frames_indices, dtype=float) / fps) if fps > 0 else np.array([])
    motion_per_frame = np.insert(motions, 0, 0.0) if motions.size > 0 else np.zeros(len(frame_timestamps))
    
    features = {
        "overall_motion": {"mean": safe_float(np.mean(motions)) if motions.size > 0 else 0.0,
                           "std": safe_float(np.std(motions)) if motions.size > 1 else 0.0},
        "frame_by_frame_motion": {"timestamps": frame_timestamps.tolist(),
                                   "motion_values": motion_per_frame.tolist()}
    }
    logging.info(f"映像解析完了 | time={time.time() - t0:.2f}s frames={len(all_landmarks)}")
    return {
        "frames_processed": len(all_landmarks), "fps": safe_float(fps),
        "duration_sec": safe_float(total_frames / fps) if fps > 0 else None,
        "features": features
    }


def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    if config.ensure_ffmpeg:
        for exe in ["ffmpeg", "ffprobe"]:
            if not shutil.which(exe): logging.warning(f"{exe} が見つかりません。")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Whisper start | model={WHISPER_MODEL_NAME} device={device} file={audio_path}")
    model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    fp16 = (device == "cuda")
    result = model.transcribe(audio_path, verbose=False, fp16=fp16, word_timestamps=True)
    logging.info(f"文字起こし完了 | time={time.time() - t0:.2f}s")
    return {
        "full_text": result.get("text", ""), "segments": result.get("segments", []),
        "language": result.get("language"), "model": WHISPER_MODEL_NAME, "device": device
    }

def _combine_words_into_segments(whisper_segments: List[Dict], max_chars: int, max_seconds: float) -> List[Dict]:
    """Whisperの単語単位のタイムスタンプを、意味のある文単位に結合する"""
    if not whisper_segments:
        return []

    combined_segments = []
    current_segment_text = ""
    current_segment_start = None
    last_word_end = None

    for seg in whisper_segments:
        for word_info in seg.get("words", []):
            word_text = word_info["word"].strip()
            word_start = word_info["start"]
            word_end = word_info["end"]

            if current_segment_start is None:
                current_segment_start = word_start

            current_segment_text += word_text + " "
            last_word_end = word_end

            if len(current_segment_text) >= max_chars or (last_word_end - current_segment_start) >= max_seconds:
                combined_segments.append({
                    "text": current_segment_text.strip(),
                    "start": current_segment_start,
                    "end": last_word_end
                })
                current_segment_text = ""
                current_segment_start = None

    if current_segment_text:
        combined_segments.append({
            "text": current_segment_text.strip(),
            "start": current_segment_start,
            "end": last_word_end
        })
    
    return combined_segments

def _get_vision_summary_for_segment(segment: Dict, vision_analysis: Dict) -> Dict:
    """音声セグメントの時間範囲に対応する映像データを抽出し、要約する"""
    start, end = segment["start"], segment["end"]
    fb_motion = vision_analysis.get("features", {}).get("frame_by_frame_motion", {})
    timestamps, motions = np.array(fb_motion.get("timestamps", [])), np.array(fb_motion.get("motion_values", []))
    
    indices = np.where((timestamps >= start) & (timestamps < end))
    segment_motions = motions[indices]
    
    if segment_motions.size == 0:
        return {"mean_motion": None, "std_motion": None, "note": "No motion data for this segment."}
    
    return {
        "mean_motion": safe_float(np.mean(segment_motions)),
        "std_motion": safe_float(np.std(segment_motions)),
        "max_motion": safe_float(np.max(segment_motions)) if segment_motions.size > 0 else None,
        "min_motion": safe_float(np.min(segment_motions)) if segment_motions.size > 0 else None,
        "note": f"Analyzed {segment_motions.size} frames."
    }

# 各専門家エージェントとメタデータ計算 
class EvaluationReport(BaseModel):
    score: Optional[float] = Field(None, description="スコア値")
    score_type: Literal["5point", "0to1"] = Field("5point", description="評価スケール")
    reason: str = Field(..., description="評価理由")
    
class SincerityReport(BaseModel):
    score: Optional[float] = Field(None, description="統合スコア")
    score_type: Literal["5point", "0to1"] = Field("5point", description="評価スケール")
    reason: str = Field(..., description="統合理由")
    sincerity_text_report: EvaluationReport = Field(..., description="言語分析からのレポート")
    sincerity_multimodal_report: EvaluationReport = Field(..., description="マルチモーダル分析からのレポート")

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _run_evaluation_agent(prompt: str, structured_llm) -> EvaluationReport:
    return structured_llm.invoke(prompt)

def _create_evaluation_agent_runner(prompt_template: str):
    def runner(text: str, structured_llm, **kwargs) -> EvaluationReport:
        prompt = prompt_template.format(text=text, **kwargs)
        try:
            return _run_evaluation_agent(prompt, structured_llm)
        except Exception as e:
            logging.exception(f"LLMエージェントの実行に失敗（リトライ上限超過）: {e}")
            return EvaluationReport(score=None, score_type="5point", reason="LLM評価がリトライ上限を超えました")
    return runner

# Big Five評価エージェント (全文で評価)
evaluate_openness_agent = _create_evaluation_agent_runner(
    "あなたは開放性（新しい経験への開放性）を分析する専門家です。以下の発言内容について、この特性の観点のみで1〜5点の5段階評価をしてください。"
    "創造性や好奇心、新しいアイデアへの関心が感じられるかを評価し、スコアと理由を構造化データとして出力してください。\n\n発言内容:\n---\n{text}"
)
evaluate_conscientiousness_agent = _create_evaluation_agent_runner(
    "あなたは誠実性（勤勉さや自己規律）を分析する専門家です。以下の発言内容について、この特性の観点のみで1〜5点の5段階評価をしてください。"
    "計画性、責任感、目的意識が感じられるかを評価し、スコアと理由を構造化データとして出力してください。\n\n発言内容:\n---\n{text}"
)
evaluate_extraversion_agent = _create_evaluation_agent_runner(
    "あなたは外向性（社交性や積極性）を分析する専門家です。以下の発言内容について、この特性の観点のみで1〜5点の5段階評価をしてください。"
    "活発さ、社交性、自信が感じられるかを評価し、スコアと理由を構造化データとして出力してください。\n\n発言内容:\n---\n{text}"
)
evaluate_agreeableness_agent = _create_evaluation_agent_runner(
    "あなたは協調性（他者への配慮や優しさ）を分析する専門家です。以下の発言内容について、この特性の観点のみで1〜5点の5段階評価をしてください。"
    "他者への共感、協力的な姿勢、友好的な態度が感じられるかを評価し、スコアと理由を構造化データとして出力してください。\n\n発言内容:\n---\n{text}"
)
evaluate_neuroticism_agent = _create_evaluation_agent_runner(
    "あなたは神経症傾向（ネガティブな感情の不安定さ）を分析する専門家です。以下の発言内容について、この特性の観点のみで1〜5点の5段階評価をしてください。"
    "不安、怒り、憂鬱、ストレスに対する脆弱性が感じられるかを評価し、スコアと理由を構造化データとして出力してください。\n\n発言内容:\n---\n{text}"
)
# 信頼性(言語)エージェント - 全文で評価
evaluate_sincerity_text_agent = _create_evaluation_agent_runner(
    "あなたは人間の誠実さを見抜く言語専門家です。以下の発言内容について、論理的な一貫性や、発言内容自体に嘘偽りがないかを判断し、1〜5点の5段階で評価してください。\n\n発言内容:\n---\n{text}"
)
# 信頼性(マルチモーダル)エージェント - 全文とセグメント映像データで評価
evaluate_sincerity_multimodal_agent = _create_evaluation_agent_runner(
    "あなたは人間の誠実さを見抜くマルチモーダル専門家です。以下の発話全体の文脈と、各セグメントにおける表情・動き（要約された映像データ）を考慮し、誠実性の観点のみで1〜5点の5段階評価をしてください。\n"
    "特に、発話内容と表情・動きの乖離を重視してください。\n\n"
    "全文:\n---\n{full_text}\n---\n\n"
    "各セグメントの要約映像データ:\n---\n{vision_summary_str}\n---"
)

# 信頼性スコアの統合関数は削除
def calculate_reliability_metrics(overall_reports: Dict, sincerity_multimodal_scores: List[Optional[float]]) -> Dict:
    def normalize_score(report: Union[dict, EvaluationReport, SincerityReport, None]) -> Optional[float]:
        if report is None:
            return None
        
        score = None
        if isinstance(report, (EvaluationReport, SincerityReport)):
            score = report.score
        elif isinstance(report, dict):
            score = report.get("score")
        
        val = safe_float(score)
        return val / 5.0 if val is not None else None

    big5_scores = {k: normalize_score(v) for k, v in overall_reports.items() if k != "sincerity_text" and k != "sincerity_multimodal"}
    
    sincerity_text_report = overall_reports.get("sincerity_text")
    sincerity_multimodal_report = overall_reports.get("sincerity_multimodal")
    
    sincerity_text_score_norm = normalize_score(sincerity_text_report)
    sincerity_multimodal_score_norm = normalize_score(sincerity_multimodal_report)
    
    discrepancy_score = None
    if sincerity_text_score_norm is not None and sincerity_multimodal_score_norm is not None:
        discrepancy_score = abs(sincerity_text_score_norm - sincerity_multimodal_score_norm)

    valid_multimodal_scores = [s for s in sincerity_multimodal_scores if s is not None]
    
    reliability = {
        "big5_scores_std": safe_float(np.nanstd([s for s in big5_scores.values() if s is not None])) if any(s is not None for s in big5_scores.values()) else None,
        "sincerity_multimodal_scores_std": safe_float(np.nanstd(valid_multimodal_scores)) if len(valid_multimodal_scores) > 1 else None,
        "sincerity_multimodal_scores_min": safe_float(np.nanmin(valid_multimodal_scores)) if len(valid_multimodal_scores) > 0 else None,
        "sincerity_multimodal_scores_max": safe_float(np.nanmax(valid_multimodal_scores)) if len(valid_multimodal_scores) > 0 else None,
        "sincerity_discrepancy_score": safe_float(discrepancy_score)
    }
    
    return reliability

# ======================== 4. メイン処理（オーケストレーター） ========================
def main():
    """スクリプト全体のエントリーポイント。"""
    run_start = time.time()
    error_log = []
    
    parser = argparse.ArgumentParser(description="映像と音声からマルチモーダルな評価JSONを生成します。")
    parser.add_argument("video_path", type=str, help="入力ビデオファイルのパス")
    parser.add_argument("audio_path", type=str, help="入力オーディオファイルのパス")
    parser.add_argument("-o", "--output", type=str, help="出力先JSONファイル名（指定がなければ自動生成）")
    args = parser.parse_args()

    try:
        video_p, audio_p = Path(args.video_path), Path(args.audio_path)
        _validate_inputs(video_p, audio_p)
    except (FileNotFoundError, ValueError) as e:
        logging.critical(f"入力ファイルの検証エラー: {e}"); error_log.append(f"Input validation error: {e}"); return
    
    output_path = args.output
    if output_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"result_{ts}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # ChatOpenAIの引数を修正 (modelとtimeoutが適切)
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0.1,
        timeout=60
    )
    structured_llm_eval = llm.with_structured_output(EvaluationReport)
    
    try:
        vision_analysis = analyze_face_landmarks(args.video_path)
        whisper_output = transcribe_audio(args.audio_path)
        
        full_text = whisper_output.get("full_text", "")
        if not full_text:
            logging.warning("文字起こしされたテキストがありません。処理を終了します。")
            return
            
        segment_config = config.segmentation
        combined_segments = _combine_words_into_segments(
            whisper_output.get("segments", []), 
            max_chars=int(segment_config.get("max_chars", 120)), 
            max_seconds=float(segment_config.get("max_seconds", 15.0))
        )

        # 1. Big Fiveと信頼性(言語)を評価
        overall_reports = {}
        with Progress() as progress:
            task = progress.add_task("Evaluating Big Five & Sincerity(Text)...", total=6)
            
            big5_agents = {
                "openness": evaluate_openness_agent, "conscientiousness": evaluate_conscientiousness_agent,
                "extraversion": evaluate_extraversion_agent, "agreeableness": evaluate_agreeableness_agent,
                "neuroticism": evaluate_neuroticism_agent
            }
            for name, agent in big5_agents.items():
                try:
                    overall_reports[name] = agent(full_text, structured_llm_eval)
                except Exception as e:
                    logging.exception(f"Big Five評価({name})に失敗しました。")
                    error_log.append(f"Big Five evaluation failed for {name}: {e}")
                    overall_reports[name] = EvaluationReport(score=None, score_type="5point", reason=f"評価失敗: {e}")
                finally:
                    progress.advance(task)
            
            try:
                sincerity_text_report = evaluate_sincerity_text_agent(full_text, structured_llm_eval)
                overall_reports["sincerity_text"] = sincerity_text_report
            except Exception as e:
                logging.exception("言語誠実性評価に失敗しました。")
                error_log.append(f"Text sincerity evaluation failed: {e}")
                overall_reports["sincerity_text"] = EvaluationReport(score=None, score_type="5point", reason=f"評価失敗: {e}")
            progress.advance(task)
            
        # 2. 信頼性(マルチモーダル)を各セグメントで評価し、レポートを収集
        sincerity_multimodal_reports = []
        vision_summaries_for_report = []
        if combined_segments:
            with Progress() as progress:
                task = progress.add_task("Evaluating Sincerity(Multimodal) per Segment...", total=len(combined_segments))
                for seg in combined_segments:
                    text = seg.get("text", "")
                    vision_summary = _get_vision_summary_for_segment(seg, vision_analysis)
                    
                    vision_summary_str = f"Mean Motion: {vision_summary.get('mean_motion')}, Std Dev: {vision_summary.get('std_motion')}, Max Motion: {vision_summary.get('max_motion')}"
                    
                    try:
                        report = evaluate_sincerity_multimodal_agent(text=text, structured_llm=structured_llm_eval,
                                                                      full_text=full_text, vision_summary_str=vision_summary_str)
                    except Exception as e:
                        logging.exception("マルチモーダル誠実性評価に失敗しました。")
                        error_log.append(f"Sincerity(MM) segment evaluation failed: {e}")
                        report = EvaluationReport(score=None, score_type="5point", reason="評価失敗")
                        
                    sincerity_multimodal_reports.append(report)
                    vision_summaries_for_report.append({
                        "start": seg["start"], "end": seg["end"],
                        "text": text, "vision_summary": vision_summary,
                        "sincerity_report": to_dict(report)
                    })
                    progress.advance(task)

        # 3. 信頼性(マルチモーダル)スコアを平均
        multimodal_scores = [r.score for r in sincerity_multimodal_reports if r.score is not None]
        multimodal_report_avg = EvaluationReport(
            score=safe_float(np.mean(multimodal_scores)) if multimodal_scores else None,
            score_type="5point",
            reason="マルチモーダル評価はセグメントごとの評価の平均です。"
        )
        overall_reports["sincerity_multimodal"] = multimodal_report_avg
        
        # 修正済み関数を呼び出し
        reliability_metrics = calculate_reliability_metrics(overall_reports, multimodal_scores)
        
        def _norm(report: Union[EvaluationReport, dict, None]) -> Optional[float]:
            if report is None:
                return None
            score = report.score if isinstance(report, EvaluationReport) else report.get("score") if isinstance(report, dict) else None
            val = safe_float(score)
            return val / 5.0 if val is not None else None

        weighted_scores = [
            (s, WEIGHTS.get(name))
            for name, r in overall_reports.items()
            if (s := _norm(r)) is not None and WEIGHTS.get(name) is not None
        ]
        
        total_w = sum(w for s, w in weighted_scores if s is not None and w is not None)
        
        overall_score = None
        if total_w is not None and total_w > 0:
            numerator = sum(s * w for s, w in weighted_scores if s is not None and w is not None)
            if numerator is not None:
                overall_score = safe_float(numerator / total_w)

    except Exception as e:
        logging.critical(f"処理中に予期せぬエラーが発生しました: {e}", exc_info=True); error_log.append(f"Unexpected error: {e}"); return
        
    run_meta = {
        "run_id": str(uuid.uuid4()),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "duration_sec": round(time.time() - run_start, 3),
        "error_log": error_log
    }

    result = {
        "schema_version": "12.8-final",
        "run_meta": run_meta,
        "versions": {
            "python": sys.version, "opencv": cv2.__version__, "mediapipe": getattr(mp, "__version__", "N/A"),
            "torch": torch.__version__, "whisper": getattr(whisper, "__version__", "N/A"),
            "langchain": langchain.__version__
        },
        "source_media": {"video_path": args.video_path, "audio_path": args.audio_path},
        "analysis_summary": {"overall_score": overall_score, "reliability_metrics": reliability_metrics},
        "overall_evaluations": {k: to_dict(v) for k, v in overall_reports.items()},
        "segment_data": vision_summaries_for_report
    }
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"マルチエージェントによる統合評価を保存しました → {output_path}")
    except Exception as e:
        logging.error(f"結果の保存に失敗しました: {output_path}\n{e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("\n処理がユーザーによって中断されました。")
        sys.exit(130)