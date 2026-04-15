import torch
import torch.nn.functional as F
from conch.open_clip_custom import get_tokenizer

def safe_tokenize(tokenizer, texts):
    # Bypass broken conch.tokenize
    tokens = tokenizer(texts, 
                       padding='max_length', 
                       max_length=127, 
                       truncation=True, 
                       return_tensors='pt')
    # conch expects 128 tokens, where the 128th is a pad placeholder
    input_ids = F.pad(tokens['input_ids'], (0, 1), value=tokenizer.pad_token_id)
    return input_ids

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    texts = ["A sample pathology caption", "Another one"]
    tokens = safe_tokenize(tokenizer, texts)
    print(f"Safe tokens shape: {tokens.shape}")
    assert tokens.shape == (2, 128)
    print("TOKENIZATION VERIFIED")
