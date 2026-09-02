"""历史视觉分析实现，仅供迁移对照；当前实现位于 src/deep_sea_explorer/。"""

import os
import cv2
import numpy as np
import tempfile
import threading
import time
import json
import logging
import queue
import base64
import re
os.environ["TRANSFORMERS_ALLOW_UNSAFE_TORCH_LOAD"] = "1"
import torch
from io import BytesIO
from datetime import datetime
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from modelscope import StableDiffusionPipeline
from rag_processor import RAGProcessor
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_analysis_")
        self.frame_buffer, self.buffer_lock = {}, threading.Lock()
        
        # 新增：显存/模型推理锁，防止多线程冲突
        self.inference_lock = threading.Lock()
        
        self.memo_interval, self.last_push_time = 3, 0
        self.model, self.processor = None, None
        self._load_model()
        self.rag = RAGProcessor()
        self._init_rag_data()
        self.img_pipe = None
        self._load_image_model()
        self.session_env_stats, self.stats_lock = {}, threading.Lock()
        self.session_cumulative_data = {}
        self.last_memo_data, self.memo_queue = {}, queue.Queue()
        self.is_processing_user_question = False
        self.last_analyzed_video = {}
        
        self.memo_thread = threading.Thread(target=self._memo_generation_worker, daemon=True)
        self.memo_thread.start()
        self.SYSTEM_PROMPT = "你是深海探测专用AI系统，专门分析深海视频和相关文档资料。你具备丰富的海洋生物学、地质学和深海探测技术知识。"
        # 语义相似度模型
        self.sim_model = SentenceTransformer(
            "/data/hmy/Semantic_comparison/GTE-Multilingual-Base", 
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 每个 session 保存上一条 memo embedding
        self.last_memo_embedding = {}

    def _init_rag_data(self):
        try:
            pdf_files = [os.path.join(r, f) for r, d, fs in os.walk(".") for f in fs if f.endswith('.pdf')]
            if pdf_files:
                for pdf in pdf_files: self.rag.add_pdf(pdf)
                self.rag.build_index()
        except Exception as e: logger.error(f"RAG初始化失败: {e}")

    def _should_use_rag(self, question: str) -> bool:
        keywords = ["文档", "报告", "资料", "记录", "数据", "信息", "内容", "详细", "具体", "探测", "发现", "生物", "温度", "深度", "样本", "设备", "任务", "航次", "什么", "多少", "如何", "哪些", "怎样", "介绍", "说明", "描述"]
        return any(k in question for k in keywords)

    def _generate_rag_prompt(self, question: str, context: str) -> str:
        return f"基于以下文档资料和视频内容回答问题：\n\n相关文档资料：\n{context}\n\n问题：{question}\n\n请结合视频内容和上述文档资料来回答问题。如果文档中有相关信息，请优先使用文档内容；如果文档中没有相关信息，则基于视频内容回答。"

    def _load_model(self):
        try:
            path = "../../Qwen/Qwen3-VL-4B-Instruct"
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            logger.info("Qwen-VL Model loaded successfully")
        except Exception as e: logger.error(f"Failed to load Qwen-VL model: {e}")

    def _load_image_model(self):
        try:

            model_path = "/data/hmy/text-to-image/stable-diffusion-v1-5"

            self.img_pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            )

            if torch.cuda.is_available():
                self.img_pipe = self.img_pipe.to("cuda")
            else:
                self.img_pipe = self.img_pipe.to("cpu")

            logger.info("Stable Diffusion v1.5 loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load SD 1.5 model: {e}")
            self.img_pipe = None



    def _update_and_get_cumulative_stats(self, session_id, category, current_items):
        with self.stats_lock:
            if session_id not in self.session_cumulative_data: self.session_cumulative_data[session_id] = {"bio": {}, "env": {}}
            summary = self.session_cumulative_data[session_id][category]
            if category == "env" and len(current_items) > 0: current_items = [current_items[0]]
            for item in current_items:
                name = item.get("name", "").strip()
                if not name or name in ["未知", "某种生物", "生物体"]: continue
                summary[name] = summary.get(name, 0) + (item.get("count", 1) if category == "bio" else 1)
            return [{"name": k, "count": v} for k, v in summary.items()]

    def _generate_memo_text(self, video_path):
        if not self.model: return None
        try:
            messages = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]}, {"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": "简述视频中发生了什么，画面有什么变化？"}]}]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(self.model.device)
            
            with self.inference_lock: # 加锁执行
                with torch.no_grad():
                    ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
            return self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        except Exception as e: logger.error(f"Memo gen failed: {e}"); return None

    def _run_image_generation(self, prompt):
        if not self.img_pipe:
            return None, "生图模型未加载"

        try:
            image = self.img_pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry, deformed",
                width=512, #分辨率
                height=512, #分辨率
                num_inference_steps=25
            ).images[0]

            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)

            return base64.b64encode(buffered.getvalue()).decode("utf-8"), None

        except Exception as e:
            logger.error(f"Image Gen Error: {e}")
            return None, str(e)

    def _is_similar_memo(self, session_id: str, new_text: str, threshold: float = 0.85) -> bool:
        """
        使用 GTE-Multilingual 连续向量进行相似度判断
        """
        if not new_text:
            return False

        # 生成密集嵌入向量 (Dense Embedding)
        # GTE 模型会自动处理推理逻辑
        new_emb = self.sim_model.encode([new_text])[0]

        # 手动执行归一化（确保点积即为余弦相似度）
        norm = np.linalg.norm(new_emb, ord=2)
        if norm > 0:
            new_emb = new_emb / norm

        last_emb = self.last_memo_embedding.get(session_id)

        # 第一条 memo：记录向量并推送
        if last_emb is None:
            self.last_memo_embedding[session_id] = new_emb
            return False

        # 计算相似度得分 (点积运算)
        similarity = float(new_emb @ last_emb.T)

        # GTE 模型精度较高，如果相似度超过阈值则不推送
        if similarity >= threshold:
            return True

        # 不相似，更新缓存并推送
        self.last_memo_embedding[session_id] = new_emb
        return False


    def _safe_json_from_text(self, text: str):
        if not text: return None
        try:
            text = text.strip()
            try: return json.loads(text)
            except: pass
            m = re.search(r"\{[\s\S]*\}", text)
            return json.loads(m.group(0)) if m else None
        except: return None

    def _llm_decide_capture_and_stats(self, image_path: str):
        if not self.model or not image_path: return None
        # 恢复完整的核心提示词
        prompt = (
            "你现在是深海科考队的首席科学家。请对这张单帧影像进行“科研价值评估”，并输出【严格 JSON】。\n"
            "判定准则：1) is_deepsea：出现海底/潜器等即为 true；2) is_typical：存在明确生物或环境目标即为 true。\n"
            "3) category：主体是清晰生物选 'bio'，否则选 'env'。\n"
            "4) organisms：仅当 bio 时填写，包含 name 和 count（不准用未知）。\n"
            "5) env_features：仅当 env 时填写，包含 name 和 count（固定填 1）。\n"
            "输出格式：{\"is_deepsea\":true,\"is_typical\":true,\"category\":\"bio\",\"description\":\"...\",\"organisms\":[{\"name\":\"铠甲虾\",\"count\":3}],\"env_features\":[]}"
        )
        try:
            msgs = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]}, {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(self.model.device)
            
            with self.inference_lock: # 加锁执行
                with torch.no_grad():
                    ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
            return self._safe_json_from_text(self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0])
        except Exception as e: logger.error(f"LLM capture error: {e}"); return None

    def _iterate_frames(self, video_path, stride=1):
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None: break
            if stride <= 1 or (idx % stride == 0): yield idx, frame
            idx += 1
        cap.release()

    def _frame_to_data_uri_and_file(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok: return None, None
        img_bytes = buf.tobytes()
        data_uri = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")
        img_path = os.path.join(self.temp_dir, f"frm_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        with open(img_path, "wb") as f: f.write(img_bytes)
        return img_path, data_uri

    def _analyze_video_segment_fast(self, session_id, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None: return None
        img_path, data_uri = self._frame_to_data_uri_and_file(frame)
        cap_info = self._llm_decide_capture_and_stats(img_path)
        try: os.remove(img_path)
        except: pass
        if not cap_info or not (cap_info.get("is_deepsea") and cap_info.get("is_typical")): return None
        cat, desc = cap_info.get("category", "env"), cap_info.get("description", "")
        if cat == "bio":
            raw = cap_info.get("organisms", []) or []
            stats = self._update_and_get_cumulative_stats(session_id, "bio", raw)
            return {"type": "bio", "image": data_uri, "description": desc, "organisms": stats, "env_features": []}
        else:
            raw = cap_info.get("env_features", []) or []
            stats = self._update_and_get_cumulative_stats(session_id, "env", raw)
            return {"type": "env", "image": data_uri, "description": desc, "organisms": [], "env_features": stats}

    def _memo_generation_worker(self):
        while True:
            try:
                # 核心：如果正在处理用户问题或生成报告，后台线程让出 GPU
                if self.is_processing_user_question: 
                    time.sleep(1)
                    continue
                    
                with self.buffer_lock: active_sessions = list(self.frame_buffer.keys())
                for sid in active_sessions:
                    v_path = self.get_latest_video(sid)
                    if not v_path or self.last_analyzed_video.get(sid) == v_path: continue
                    self.last_analyzed_video[sid] = v_path
                    m_txt = self._generate_memo_text(v_path)
                    if not m_txt: continue
                    # 语义相似度判断
                    if self._is_similar_memo(sid, m_txt): continue
                    c_data = self._analyze_video_segment_fast(sid, v_path)
                    self.memo_queue.put({"timestamp": datetime.now().strftime("%H:%M:%S"), "content": m_txt, "session_id": sid, "capture": c_data})
                time.sleep(1)
            except Exception as e: logger.error(f"Worker error: {e}"); time.sleep(2)

    def stream_process_question(self, session_id, question, video_path):
        if not self.model: yield json.dumps({"type": "error", "text": "模型未加载"}) + "\n"; return
        triggers = ["画", "生成图片", "create image", "generate image", "生成", "生成一张","Generate","generate"]
        if any(t in question for t in triggers):
            prompt = question
            for t in triggers: prompt = prompt.replace(t, "")
            yield json.dumps({"type": "chunk", "text": f"正在为您构想图..."}) + "\n"
            img_b64, err = self._run_image_generation(prompt)
            if img_b64: yield json.dumps({"type": "image", "content": img_b64, "prompt": prompt}) + "\n"
            else: yield json.dumps({"type": "error", "text": f"生图失败: {err}"}) + "\n"
            return
        
        final_q, use_rag = question, False
        if self._should_use_rag(question) and self.rag.index is not None:
            ctx = self.rag.get_context(question)
            if ctx: final_q = self._generate_rag_prompt(question, ctx); use_rag = True
            
        try:
            msgs = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]}, {"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": final_q}]}]
            inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(self.model.device)
            
            with self.inference_lock: # 加锁执行
                streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
                threading.Thread(target=self.model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=512, do_sample=True, temperature=0.7)).start()
                f_text = ""
                for new_text in streamer:
                    f_text += new_text; yield json.dumps({"type": "chunk", "text": new_text}) + "\n"
                yield json.dumps({"type": "final", "text": f_text}) + "\n"
        except Exception as e: yield json.dumps({"type": "error", "text": str(e)}) + "\n"

    def process_frames(self, session_id, frames_data):
        s_dir = os.path.join(self.temp_dir, session_id); os.makedirs(s_dir, exist_ok=True)
        with self.buffer_lock:
            if session_id not in self.frame_buffer: self.frame_buffer[session_id] = {'videos': [], 'last_update': datetime.now()}
        v_path = os.path.join(s_dir, f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        try:
            first = cv2.imdecode(np.frombuffer(frames_data[0], np.uint8), cv2.IMREAD_COLOR)
            if first is None: return None
            h, w = first.shape[:2]
            out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
            for fd in frames_data:
                f = cv2.imdecode(np.frombuffer(fd, np.uint8), cv2.IMREAD_COLOR)
                if f is not None: out.write(f)
            out.release()
            with self.buffer_lock: self.frame_buffer[session_id]['videos'] = [v_path]
            return v_path
        except: return None

    def get_latest_video(self, session_id):
        with self.buffer_lock: return self.frame_buffer.get(session_id, {}).get('videos', [None])[-1]

    def get_memos(self):
        m = []
        while not self.memo_queue.empty(): m.append(self.memo_queue.get())
        return m
