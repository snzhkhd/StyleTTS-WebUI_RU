import os
import subprocess
import gradio as gr
from datetime import datetime
import multiprocessing
import webbrowser
import socket

def get_training_config(voice):
    config_path = os.path.join("training", voice, f"{voice}_config.yml")
    return config_path

def start_training_proxy(voice, progress=gr.Progress(track_tqdm=True)):
    from styletts2.train_finetune_accelerate import main as run_train
    config_path = get_training_config(voice)
    run_train(config_path)
    return "Training Complete!"

def launch_tensorboard_proxy():
    port = 6006
    if is_port_in_use(port):
        gr.Warning(f"Port {port} is already in use. Skipping TensorBoard launch.")
    else:
        subprocess.Popen(["launch_tensorboard.bat"], shell=True)
        time.sleep(1)
    webbrowser.open(f"http://localhost:{port}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) == 0

def get_folder_list(root):
    return [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

if __name__ == "__main__":
    train_list = get_folder_list(root="training")

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Training"):
                with gr.Tabs():
                    with gr.TabItem("Run Training"):
                        with gr.Row():
                            with gr.Column():
                                training_voice_name = gr.Dropdown(
                                    label="Voice Name",
                                    choices=train_list,
                                    value=train_list[0] if train_list else None
                                )
                                refresh_available_config_button_2 = gr.Button(value="Refresh Available")
                            with gr.Column():
                                training_console = gr.Textbox(label="Training Console")
                                start_train_button = gr.Button(value="Start Training")
                        with gr.Row():
                            launch_tensorboard_button = gr.Button(value="Launch Tensorboard")
                            
                        start_train_button.click(
                            start_training_proxy,
                            inputs=[training_voice_name],
                            outputs=[training_console]
                        )
                        launch_tensorboard_button.click(launch_tensorboard_proxy)
                        refresh_available_config_button_2.click(
                            lambda: get_folder_list(root="training"),
                            outputs=[training_voice_name]
                        )

    webui_port = None
    while webui_port is None:
        for i in range(7860, 7865):
            if not is_port_in_use(i):
                webui_port = i
                break
    webbrowser.open(f"http://localhost:{webui_port}")
    demo.launch(server_port=webui_port)