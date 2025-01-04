import gradio as gr
import json
from jinja2 import Template


def get_data(choice):
    with open("./Raw_Data/" + mapper[choice], "r") as f:
        return json.load(f)


def process_chat(conversations, prompt_type, clean_sharegpt):
    history = []
    for sample in conversations:
        user_prompt = sample["instruction"]
        if prompt_type == "ShareGPT" and clean_sharegpt:
            user_prompt = (
                user_prompt.removeprefix("USER:").removesuffix("ASSISTANT:").strip()
            )
        history.append([user_prompt, sample["output"]])
    return history


def render_html(json_data, processed_chat):
    template_string = """<html>
<head>
        <title>LLM Benchmark Report for: {{ model_name }}</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet"> 
            <style>
        body {
            font-family: 'Roboto', sans-serif;
            padding: 20px;
            margin: 0;
            background-color: #f4f7f9;
            color: #333;
        }
        h1, h2 {
            color: #2C3E50;
        }
        pre {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        p {
            font-size: 0.95em;
            color: #7f8c8d;
        }
        strong {
            color: #2C3E50;
        }
        hr {
            margin: 40px 0;
            border: none;
            border-top: 1px solid #e0e0e0;
        }
        header, footer {
            background-color: #2C3E50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        footer {
            margin-top: 20px;
        }
        .info-label {
            font-weight: bold;
            color: #3498db;
            margin-right: 10px;
        }
        .benchmark-section {
            margin-bottom: 20px;
        }
        .info-section p {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>LLM Benchmark Report for: {{ model_name }}</h1>
    
<ul>
    <li><strong>Total Prompts:</strong> {{ total_prompts }}</li>
    <li><strong>Model Name:</strong> {{ model_name }}</li>
    <li><strong>Prompt Format:</strong> {{ prompt_format }}</li>
    <li><strong>Temperature:</strong> {{ temperature }}</li>
    <li><strong>Top P:</strong> {{ top_p }}</li>
    <li><strong>Presence Penalty:</strong> {{ presence_penalty }}</li>
    <li><strong>Frequency Penalty:</strong> {{ frequency_penalty }}</li>
</ul>

    
{% for item in chat %}

<div class="benchmark-section">
    <h3>Instruction {{ loop.index }}</h3>
    <pre>{{ item[0] }}
</pre>
    <h3>Response {{ loop.index }} </h3>
    <pre>{{ item[1] }}
</pre>
</div>


{% endfor %}

</body>
</html>
"""
    total_prompts = len(processed_chat)
    template = Template(template_string)
    rendered_template = template.render(
        **json_data, total_prompts=total_prompts, chat=processed_chat
    )
    return rendered_template


mapper = {
    "Nous-Capybara-7B ShareGPT": "NousResearch-Nous-Capybara-7B_September_25_2023.json",
    "Nous-Hermes-llama-2-7b Alpaca": "NousResearch-Nous-Hermes-llama-2-7b_September_25_2023.json",
    "Redmond-Puffin-13B ShareGPT": "NousResearch-Redmond-Puffin-13B_September_25_2023.json",
    "teknium-OpenHermes-13B Alpaca": "teknium-OpenHermes-13B_September_25_2023.json",
    "teknium-OpenHermes-7B Alpaca": "teknium-OpenHermes-7B_September_25_2023.json",
    "PygmalionAI-mythalion-13b Alpaca": "PygmalionAI-mythalion-13b_September_25_2023.json",
    "Nous-Hermes-llama-2-13B Alpaca": "NousResearch-Nous-Hermes-Llama2-13b_September_25_2023.json",
}

choices = list(mapper.keys())

with gr.Blocks(title="LLM-LOGBOOK") as demo:
    dropdown_menu = gr.Dropdown(choices, value=choices[0], label="Choose an LLM")
    with gr.Row():
        output_format = gr.Radio(
            ["html", "chat interface", "json"],
            value="html",
            label="output format",
        )
        clean_sharegpt = gr.Checkbox(True, label="clean sharegpt")

    @gr.render(inputs=[dropdown_menu, output_format, clean_sharegpt])
    def update_interface(choice, output_format, clean_sharegpt):
        json_data = get_data(choice)

        if output_format == "json":
            gr.JSON(json_data)
        else:
            processed_chat = process_chat(
                json_data["conversations"], json_data["prompt_format"], clean_sharegpt
            )

        if output_format == "chat interface":
            gr.Markdown(f"# {json_data['model_name']}")
            with gr.Accordion("parameters", open=False):
                gr.Markdown(f"* Total Prompts: {len(json_data['conversations'])}")
                gr.Markdown(f"* Model Name: {json_data['model_name']}")
                gr.Markdown(f"* Prompt Format: {json_data['prompt_format']}")
                gr.Markdown(f"* Temperature: {json_data['prompt_format']}")
                gr.Markdown(f"* Top P: {json_data['top_p']}")
                gr.Markdown(f"* Presence Penalty: {json_data['presence_penalty']}")
                gr.Markdown(f"* Frequency Penalty: {json_data['frequency_penalty']}")

            gr.Chatbot(processed_chat, type="tuples")
        if output_format == "html":
            gr.Markdown(render_html(json_data, processed_chat))


demo.launch(debug=True)
