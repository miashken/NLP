"""
@author: Michal Ashkenazi
T5: Text-To-Text Transfer Transformer
"""
from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig

model_name    = "t5-base"
t5_tokenizer  = T5Tokenizer.from_pretrained(model_name)
t5_model      = T5ForConditionalGeneration.from_pretrained(model_name)

# (zero shot) inference
task_prefix = "Translate English to French: "
sentence    = "How are you?"

input       = t5_tokenizer(text=task_prefix + sentence,
                           return_tensors="pt")

output_seq  = t5_model.generate(input_ids=input.input_ids)

output      = t5_tokenizer.decode(token_ids=output_seq[0],
                                  skip_special_tokens=True)

print(output)

# batched (zero-shot) inference
sentences   = ["Hello", "How are you?", "What is your name?"]

inputs      = t5_tokenizer(text=[task_prefix + sentence for sentence in sentences],
                           return_tensors="pt",
                           padding=True)

output_seq  = t5_model.generate(input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask)

output      = t5_tokenizer.batch_decode(sequences=output_seq,
                                        skip_special_tokens=True)

print(output)