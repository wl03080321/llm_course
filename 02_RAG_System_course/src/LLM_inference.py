from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import threading
from typing import List, Dict, Optional, Union
from pathlib import Path as path

class LLMInference:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", 
                 quantization_config: Optional[BitsAndBytesConfig] = None,
                 cache_dir: Optional[str] = None):
        """
        初始化 LLM 推理器
        
        Args:
            model_name: 模型名稱
            quantization_config: 量化配置
            cache_dir: 快取目錄
        """
        self.model_name = model_name
        self.quantization_config = quantization_config
        
        # 設定快取目錄
        if cache_dir is None:
            base_dir = path(__file__).parent.parent.parent
            self.cache_dir = base_dir / "cache"
        else:
            self.cache_dir = cache_dir
            
        # 初始化模型和分詞器
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """載入模型"""
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            trust_remote_code=True,
            device_map='auto',
            use_auth_token=True,
            cache_dir=self.cache_dir
        )
        print("Model loaded successfully!")
    
    def _load_tokenizer(self):
        """載入分詞器"""
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map='auto',
            use_auth_token=True,
            cache_dir=self.cache_dir
        )
        print("Tokenizer loaded successfully!")
        
    def generate(self,
                messages: Union[List[Dict], str], 
                max_new_tokens: int = 128, temperature: float = 0.7,
                top_k: int = 50, top_p: float = 0.95):
        """
        串流對話推理生成器 (適用於 Gradio)
        
        Args:
            input_data: 輸入資料（字串或對話列表）
            max_new_tokens: 最大新生成 token 數
            temperature: 溫度參數
            top_k: top-k 採樣參數
            top_p: top-p 採樣參數
            
        Yields:
            逐步生成的文字片段
        """
        # 處理輸入資料
        if isinstance(messages, str):
            inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]
        else:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            
        # 設定串流器
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,           # 跳過提示詞
            skip_special_tokens=True    # 跳過特殊 token
        )
        
        # 設定生成執行緒
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                streamer=streamer
            )
        )
        
        # 開始生成
        self.model.eval()
        thread.start()
        for new_text in streamer:
            if new_text is None:
                continue
            yield new_text

    def get_model_info(self) -> Dict:
        """
        獲取模型資訊
        
        Returns:
            模型資訊字典
        """
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "quantization_enabled": self.quantization_config is not None,
            "cache_dir": str(self.cache_dir)
        }
