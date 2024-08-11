import gradio as gr


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

def generate_text(input_text):
  
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

  
    attention_mask = torch.ones_like(input_ids)


    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else None


    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id

  
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=pad_token_id
    )


    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message..."),
    outputs=gr.Textbox(label="Response"),
    title="Transformer Chatbot",
    description="Chat with a transformer-based language model."
)


iface.launch()
