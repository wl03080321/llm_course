import gradio as gr
import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.path.join(base_dir, "file_temp")
# 建立檔案儲存資料夾
os.makedirs(upload_folder, exist_ok=True)

# 用來記錄最近上傳的檔案名稱清單
latest_uploaded_files = []

# 檔案上傳處理函數：儲存多個檔案
def handle_multi_upload(files):
    global latest_uploaded_files
    latest_uploaded_files = []
    image_files = []
    if files is not None:
        for file in files:
            filename = os.path.basename(file.name)
            save_path = os.path.join(upload_folder, filename)
            shutil.copy(file.name, save_path)
            latest_uploaded_files.append(filename)
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(save_path)
        result_text = f"已上傳 {len(latest_uploaded_files)} 個檔案：\n" + "\n".join(latest_uploaded_files)
        return result_text, image_files
    else:
        return "沒有檔案被上傳", []

def chatbot_reply(message, history):
    global latest_uploaded_files
    ids = len(history) + 1
    reply = f"你說了：{message}"
    if len(history) > 0:
        history[-1]["metadata"] = {"ids": ids}
    print(f"歷史紀錄：{history}")
    if latest_uploaded_files:
        reply += f"\n你上傳了：\n" + "\n".join(latest_uploaded_files)
    return reply

with gr.Blocks() as demo:
    gr.Markdown("## 使用者介面 + 檔案上傳 Demo")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(fn=chatbot_reply,
                                       type="messages")

        with gr.Column(scale=1):
            with gr.Group(elem_id="upload_scroll_area"):
                file_upload = gr.File(label="上傳檔案",
                                      file_types=[".png",".jpg",".pdf"],
                                      file_count="multiple",
                                      height=150
                                      )
                file_output = gr.Textbox(label="上傳結果",
                                         interactive=False,
                                         )
            image_gallery = gr.Gallery(label="圖片預覽") 
            file_upload.change(fn=handle_multi_upload,
                               inputs=file_upload,
                               outputs=[file_output, image_gallery])
if __name__ == "__main__":
    demo.launch()
