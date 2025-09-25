
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2", use_fast=True)
tokenizer.save_pretrained("artifacts/tokenizer")
