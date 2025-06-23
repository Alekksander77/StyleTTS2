import subprocess
import gradio as gr

from styletts2.inference import TTSModel, prepare_dataset

_current_model = None


def ui_prepare(data_dir, output):
    prepare_dataset(data_dir, output)
    return f"Saved list to {output}"


def ui_train(config, stage):
    if stage == "first":
        cmd = ["accelerate", "launch", "styletts2/train/first_stage.py", "--config_path", config]
    elif stage == "second":
        cmd = ["python", "styletts2/train/second_stage.py", "--config_path", config]
    elif stage == "finetune":
        cmd = ["python", "styletts2/train/finetune.py", "--config_path", config]
    else:
        return "Unknown stage"
    subprocess.Popen(cmd)
    return "Training started"


def ui_load(config, ckpt):
    global _current_model
    _current_model = TTSModel(config, ckpt)
    return "Model loaded"


def ui_infer(text, steps, scale):
    if _current_model is None:
        return "Model not loaded", None
    wav = _current_model.synthesize(text, diffusion_steps=int(steps), embedding_scale=float(scale))
    return "Done", (24000, wav)


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# StyleTTS2 WebUI")
        with gr.Tab("Prepare Dataset"):
            data_dir = gr.Textbox(label="LJSpeech Folder")
            out_file = gr.Textbox(value="Data/train_list.txt", label="Output list")
            prep_btn = gr.Button("Prepare")
            prep_out = gr.Textbox(label="Status")
            prep_btn.click(ui_prepare, [data_dir, out_file], prep_out)
        with gr.Tab("Train"):
            cfg = gr.Textbox(value="Configs/config.yml", label="Config path")
            stage = gr.Radio(["first", "second", "finetune"], value="first", label="Stage")
            train_btn = gr.Button("Start")
            train_out = gr.Textbox(label="Status")
            train_btn.click(ui_train, [cfg, stage], train_out)
        with gr.Tab("Inference"):
            cfg_inf = gr.Textbox(value="Configs/config.yml", label="Config path")
            ckpt = gr.Textbox(value="model.pth", label="Checkpoint")
            load_btn = gr.Button("Load")
            load_out = gr.Textbox(label="Status")
            load_btn.click(ui_load, [cfg_inf, ckpt], load_out)
            text = gr.Textbox(label="Text")
            steps = gr.Slider(1, 10, value=5, step=1, label="Diffusion steps")
            scale = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Embedding scale")
            infer_btn = gr.Button("Synthesize")
            audio_out = gr.Audio()
            infer_btn.click(ui_infer, [text, steps, scale], [load_out, audio_out])
    return demo


def main():
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
