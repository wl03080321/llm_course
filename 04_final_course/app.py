import gradio as gr
import os
import shutil
from src.RAG_System import RAGSystem
from src.LLM_inference import LLMInference
from transformers import BitsAndBytesConfig

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.path.join(base_dir, "file_temp")
os.makedirs(upload_folder, exist_ok=True)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 啟用 8-bit 量化
)
rag_system = RAGSystem(embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
llm = LLMInference(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=quantization_config,  # 使用預設配置
)

latest_uploaded_files = []

def handle_multi_upload(files):
    global latest_uploaded_files
    
    # 如果沒有檔案（取消上傳），不做任何操作，保持現有檔案
    if files is None or len(files) == 0:
        return "沒有選擇檔案"
    
    # 處理結果訊息
    result_messages = []
    new_uploaded_files = []
    
    # 獲取向量資料庫中現有的檔案列表
    existing_files = rag_system.get_all_filenames()
    
    # 處理每個上傳的檔案
    for file in files:
        filename = os.path.basename(file.name)
        save_path = os.path.join(upload_folder, filename)
        
        # 如果檔案已存在於向量資料庫中，先完全刪除所有相關的 chunks
        if filename in existing_files:
            print(f"檔案 {filename} 已存在於向量資料庫中，正在刪除所有舊的文件片段...")
            result_messages.append(f"📋 檔案 {filename} 已存在，正在覆蓋更新...")
            
            delete_result = rag_system.delete_documents_by_filename(filename)
            if delete_result["success"]:
                deleted_count = delete_result.get("deleted_count", 0)
                print(f"✓ 成功刪除檔案 {filename} 的 {deleted_count} 個文件片段")
                result_messages.append(f"  ✓ 已刪除舊版本的 {deleted_count} 個文件片段")
                
                # 從 latest_uploaded_files 中移除舊版本
                if filename in latest_uploaded_files:
                    latest_uploaded_files.remove(filename)
                    
                # 刪除舊的臨時檔案（如果存在）
                if os.path.exists(save_path):
                    os.remove(save_path)
                    print(f"  ✓ 已刪除舊的臨時檔案: {filename}")
            else:
                error_msg = delete_result.get("message", "未知錯誤")
                print(f"✗ 刪除檔案 {filename} 失敗: {error_msg}")
                result_messages.append(f"  ✗ 刪除舊版本失敗: {error_msg}")
                continue  # 跳過這個檔案，不進行後續處理
        else:
            result_messages.append(f"📋 新檔案 {filename}")
        
        # 複製新檔案到臨時資料夾
        try:
            shutil.copy(file.name, save_path)
            new_uploaded_files.append(filename)
            print(f"✓ 已複製檔案到臨時資料夾: {filename}")
        except Exception as e:
            error_msg = f"複製檔案 {filename} 失敗: {str(e)}"
            print(f"✗ {error_msg}")
            result_messages.append(f"  ✗ {error_msg}")
            continue
    
    # 使用 RAG 系統處理上傳的檔案
    if new_uploaded_files:
        print(f"開始處理 {len(new_uploaded_files)} 個檔案...")
        results = rag_system.process_uploaded_files(
            uploaded_files=new_uploaded_files,
            upload_folder=upload_folder,
            chunk_size=4192,
            chunk_overlap=200,
            add_to_vectorstore=True
        )
        
        # 更新 latest_uploaded_files（添加新檔案）
        for filename in results["processed_files"]:
            if filename not in latest_uploaded_files:
                latest_uploaded_files.append(filename)
        
        # 整合所有處理結果訊息
        final_result = []
        final_result.extend(result_messages)
        final_result.append(f"\n📊 處理結果摘要:")
        final_result.append(f"成功處理: {len(results['processed_files'])} 個檔案")
        
        # 顯示成功處理的檔案詳情
        for filename in results["processed_files"]:
            details = results["file_details"].get(filename, {})
            chunks_count = details.get('chunks', 0)
            final_result.append(f"  ✓ {filename} - 新增 {chunks_count} 個文字區塊")
        
        # 顯示處理失敗的檔案
        if results["failed_files"]:
            final_result.append(f"\n❌ 處理失敗: {len(results['failed_files'])} 個檔案")
            for failed in results["failed_files"]:
                final_result.append(f"  ✗ {failed['filename']} - 錯誤: {failed['error']}")
        
        # 顯示總計資訊
        total_docs = results.get('total_documents', 0)
        final_result.append(f"\n📈 總共新增了 {total_docs} 個文字區塊到向量資料庫")
        
        return "\n".join(final_result)
    else:
        # 如果沒有檔案被處理，返回處理過程中的訊息
        if result_messages:
            result_messages.append("\n❌ 沒有檔案成功處理")
            return "\n".join(result_messages)
        else:
            return "沒有檔案被處理"

def chatbot_reply(message, history, system_prompt, max_token, temperature, top_p, top_k):
    global latest_uploaded_files
    
    try:
        # 建構對話訊息
        messages = []
        
        # 如果有 system prompt，加入系統訊息
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        
        # 加入歷史對話 (使用 Gradio 提供的 history)
        for msg in history:
            messages.append(msg)
        
        # 檢查 RAG 系統中是否有檔案
        actual_files = rag_system.get_all_filenames()
        
        if actual_files:
            # 確保有系統訊息
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0]["content"] += """\n\n
            你將會收到由使用者的問題並透過檢索文字後的結果，檢索的內容會放入到<CONTENT>標籤中。
            使用者的問題會放在<QUESTION>標籤中。
            請根據檢索到的內容回答問題。
            """
            
            retrieved_docs = rag_system.query_and_retrieve(message, k=3, return_scores=True)
            
            if retrieved_docs:
                # 整理檢索到的內容
                context_parts = []
                for i, doc in enumerate(retrieved_docs):
                    context_parts.append(f"參考資料 {i+1} (相似度: {doc['score']:.3f}):\n{doc['content']}")
                context = "\n\n".join(context_parts)
            else:
                context = ""
                
            rag_prompt = f"""
以下是檔案中相關資料：
<CONTENT>
{context}
</CONTENT>

<QUESTION>
{message}
</QUESTION>
"""
            messages.append({"role": "user", "content": rag_prompt})
            
        else:
            # 沒有上傳檔案：一般對話
            messages.append({"role": "user", "content": message})
        
        # 使用 LLM 生成回應（串流顯示）
        for partial_response in llm.generate(
            messages, 
            max_new_tokens=max_token,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        ):
            yield partial_response
        
    except Exception as e:
        error_msg = f"發生錯誤："
        print(f"Error in chatbot_reply: {e}")
        yield error_msg

def clear_vector_database():
    """清空向量資料庫的所有資料"""
    try:
        # 獲取所有檔案名稱並逐一刪除
        filenames = rag_system.get_all_filenames()
        if filenames:
            for filename in filenames:
                result = rag_system.delete_documents_by_filename(filename)
                if result["success"]:
                    print(f"已刪除檔案 {filename} 的 {result['deleted_count']} 個文件")
                else:
                    print(f"刪除檔案 {filename} 失敗: {result['message']}")
        print("已清空向量資料庫")
    except Exception as e:
        print(f"清空向量資料庫時發生錯誤: {e}")

def clear_temp_files():
    """清空臨時檔案資料夾"""
    try:
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("已清空臨時檔案")
    except Exception as e:
        print(f"清空臨時檔案時發生錯誤: {e}")

def update_status():
    """更新向量資料庫狀態顯示"""
    global latest_uploaded_files
    
    # 從 RAG 系統獲取實際的檔案名稱
    actual_files = rag_system.get_all_filenames()
    print(f"向量資料庫中實際的檔案: {actual_files}")
    print(f"全域變數中記錄的檔案: {latest_uploaded_files}")
    
    # 同步 latest_uploaded_files 與實際的檔案列表
    latest_uploaded_files = actual_files.copy() if actual_files else []
    
    # 向量資料庫狀態
    try:
        info = rag_system.get_vectorstore_info()
        db_text = f"集合: {info.get('collection_name', 'N/A')}\n檔案數量: {info.get('total_files', 0)}\n文件片段數量: {info.get('total_documents', 0)}\n嵌入模型: {info.get('embedding_model', 'N/A')}"
    except Exception as e:
        db_text = f"無法獲取資料庫資訊: {str(e)}"
    
    return db_text

def get_uploaded_files_choices():
    """獲取可選擇刪除的檔案列表"""
    try:
        filenames = rag_system.get_all_filenames()
        print(f"從 RAG 系統獲取的檔案列表: {filenames}")
        return filenames if filenames else []
    except Exception as e:
        print(f"獲取檔案列表時發生錯誤: {e}")
        return []

def delete_selected_files(selected_files):
    """刪除選中的檔案"""
    global latest_uploaded_files
    
    if not selected_files:
        return "請選擇要刪除的檔案", get_uploaded_files_choices()
    
    results = []
    deleted_files = []
    
    # 先獲取當前實際的檔案列表，避免檔案不存在的錯誤
    current_files = rag_system.get_all_filenames()
    
    for filename in selected_files:
        # 檢查檔案是否確實存在於向量資料庫中
        if filename not in current_files:
            results.append(f"✗ {filename} - 檔案在向量資料庫中不存在")
            continue
            
        result = rag_system.delete_documents_by_filename(filename)
        if result["success"]:
            results.append(f"✓ {filename} - 已刪除 {result['deleted_count']} 個文件")
            deleted_files.append(filename)
            # 從 latest_uploaded_files 中移除
            if filename in latest_uploaded_files:
                latest_uploaded_files.remove(filename)
        else:
            results.append(f"✗ {filename} - {result['message']}")
    
    # 同時刪除臨時檔案
    for filename in deleted_files:
        temp_file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            print(f"Error deleting temp file {filename}: {e}")
    
    result_message = "檔案刪除結果：\n" + "\n".join(results)
    
    # 檢查是否所有檔案都被刪除了，如果是的話重置 latest_uploaded_files
    remaining_files = rag_system.get_all_filenames()
    if not remaining_files:
        latest_uploaded_files = []
        print("所有檔案已刪除，重置系統狀態")
    
    updated_file_choices = get_uploaded_files_choices()
    return result_message, updated_file_choices

def get_file_details():
    """獲取檔案詳細資訊"""
    try:
        filenames = rag_system.get_all_filenames()
        if not filenames:
            return "向量資料庫中沒有檔案"
        
        details = []
        total_docs = 0
        
        for filename in filenames:
            doc_count = rag_system.get_file_document_count(filename)
            details.append(f"📄 {filename}: {doc_count} 個文件")
            total_docs += doc_count
        
        details.append(f"\n總計: {len(filenames)} 個檔案，{total_docs} 個文件")
        return "\n".join(details)
        
    except Exception as e:
        return f"獲取檔案詳細資訊時發生錯誤: {str(e)}"

def clear_all_files():
    global latest_uploaded_files
    latest_uploaded_files = []
    clear_vector_database()
    clear_temp_files()
    
    db_text = update_status()
    file_details_text = get_file_details()
    file_choices = get_uploaded_files_choices()
    return "已清除所有檔案和向量資料庫", db_text, file_choices, file_details_text, None


# 建立 Gradio 頁面
with gr.Blocks(title="RAG ChatBot System") as demo:
    gr.Markdown("# 🧠 RAG ChatBot + 📁 檔案上傳系統")
    gr.Markdown("上傳 PDF 檔案後可進行基於文件的問答，未上傳檔案時可進行一般對話。")

    with gr.Row():
        # 左側邊欄：模型設定
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 模型設定")
            
            # System Prompt 設定
            system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="輸入系統提示詞 (可選)",
                lines=3,
                value=""
            )
            
            # 模型參數設定
            with gr.Group():
                gr.Markdown("#### 🎛️ 生成參數")
                max_token = gr.Slider(
                    minimum=512,
                    maximum=8192,
                    value=512,
                    step=1,
                    label="max_token",
                    info="限制詞彙候選數量 (1-8192)"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="控制生成的隨機性 (0.1-2.0)"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p",
                    info="核心採樣參數 (0.1-1.0)"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="限制詞彙候選數量 (1-100)"
                )
                
        # 中間：聊天介面
        with gr.Column(scale=2):
            # 建立聊天介面
            chatbot = gr.ChatInterface(
                fn=chatbot_reply,
                additional_inputs=[
                    system_prompt,
                    max_token,
                    temperature,
                    top_p,
                    top_k
                ],
                title="🤖 智能問答助手",
                type="messages"  # 使用新的 messages 格式
            )

        # 右側邊欄：檔案上傳和向量資料庫
        with gr.Column(scale=1):
            gr.Markdown("### 📁 檔案管理")
            
            # 檔案上傳區域
            with gr.Group():
                file_upload = gr.File(
                    label="上傳檔案 (支援 PDF)",
                    file_types=[".pdf"],
                    file_count="multiple",
                    height=120
                )
            # 檔案刪除區域
            with gr.Group():
                gr.Markdown("#### 🗑️ 檔案刪除")
                files_to_delete = gr.CheckboxGroup(
                    label="選擇要刪除的檔案",
                    choices=get_uploaded_files_choices(),  # 初始化時就獲取檔案列表
                    interactive=True
                )
                
                with gr.Row():
                    delete_selected_btn = gr.Button("🗑️ 刪除選中檔案", variant="secondary", size="sm")
                
                delete_result = gr.Textbox(
                    label="刪除結果",
                    interactive=False,
                    lines=3
                )
            
            # 向量資料庫資訊
            with gr.Group():
                gr.Markdown("#### 📊 向量資料庫狀態")
                db_info = gr.Textbox(
                    label="資料庫資訊",
                    interactive=False,
                    lines=5
                )
                file_details = gr.Textbox(
                    label="檔案詳細資訊",
                    interactive=False,
                    lines=4
                )
                
                refresh_btn = gr.Button("🔄 重新整理資訊", size="sm")

            # 檔案上傳處理
            def handle_multi_upload_with_status(files):
                result_text = handle_multi_upload(files)
                db_text = update_status()
                file_details_text = get_file_details()
                # 處理完成後清除上傳區域，避免重複上傳
                return db_text, file_details_text, None, ""  # 清空刪除結果
            
            file_upload.change(
                fn=handle_multi_upload_with_status,
                inputs=file_upload,
                outputs=[db_info, file_details, file_upload, delete_result]
            ).then(
                fn=lambda: gr.CheckboxGroup(choices=get_uploaded_files_choices(), value=[]),
                outputs=files_to_delete
            )


            # 刪除選中檔案
            def delete_files_and_update(selected_files):
                result_message, updated_choices = delete_selected_files(selected_files)
                return result_message, gr.CheckboxGroup(choices=updated_choices, value=[])
            
            delete_selected_btn.click(
                fn=delete_files_and_update,
                inputs=files_to_delete,
                outputs=[delete_result, files_to_delete]
            ).then(
                fn=update_status,
                outputs=[db_info]
            ).then(
                fn=get_file_details,
                outputs=file_details
            )
            
            # 重新整理資訊按鈕
            def refresh_info():
                db_text = update_status()
                file_details_text = get_file_details()
                return db_text, file_details_text
            
            
            refresh_btn.click(
                fn=refresh_info,
                outputs=[db_info, file_details]
            )
            # 清除檔案按鈕
            with gr.Row():
                clear_files_btn = gr.Button("🗑️ 清除所有檔案", variant="secondary", size="sm")
            
            def clear_all_and_update():
                result = clear_all_files()
                return (
                    result[0],  # delete_result message
                    result[1],  # db_info
                    gr.CheckboxGroup(choices=[], value=[]),  # files_to_delete (empty)
                    result[3],  # file_details
                    None  # file_upload (clear)
                )
            
            clear_files_btn.click(
                fn=clear_all_and_update,
                outputs=[delete_result, db_info, files_to_delete, file_details, file_upload]
            )
            
    # 頁面載入時初始化狀態
    def init_status():
        print("正在初始化頁面狀態...")
        
        # 強制刷新向量資料庫狀態
        db_text = update_status()
        print(f"資料庫狀態: {db_text}")
        
        # 獲取檔案詳細資訊
        file_details_text = get_file_details()
        print(f"檔案詳細資訊: {file_details_text}")
        
        # 獲取檔案選擇列表並返回 CheckboxGroup
        file_choices = get_uploaded_files_choices()
        print(f"可選擇刪除的檔案: {file_choices}")
        
        return db_text, file_details_text, gr.CheckboxGroup(choices=file_choices, value=[])
    
    demo.load(
        fn=init_status,
        outputs=[db_info, file_details, files_to_delete]
    )
    
if __name__ == "__main__":
    demo.launch()
