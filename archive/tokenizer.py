from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
# i'm really sorry Hugging Face, love you all but I gotta learn here
chat_template = tokenizer.chat_template.replace(
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
    "You are a helpful AI assistant named Blinky with multimodal capabilities, trained by shreydan"
)
tokenizer.chat_template = chat_template
tokenizer.save_pretrained('./Blinky')
tokenizer.save_vocabulary('./Blinky')