from transformers import MarianMTModel, MarianTokenizer
import csv
import pandas as pd


def download(model_name):
    """download data for a language"""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def assign_GPU(Tokenizer_output):
    """put tensors to GPU"""
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda:0')
    # token_type_ids = Tokenizer_output['token_type_ids'].to('cuda:0')
    attention_mask = Tokenizer_output['attention_mask'].to('cuda:0')

    output = {'input_ids' : tokens_tensor, 
              # 'token_type_ids' : token_type_ids, 
              'attention_mask' : attention_mask}

    return output


def translate(texts, model, tokenizer, language):
    """Translate texts into a target language"""
    # Format the text as expected by the model
    formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
    original_texts = [formatter_fn(txt) for txt in texts]
    translated = model.generate(**assign_GPU(tokenizer(original_texts, return_tensors="pt", padding=True)))
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts


def back_translate(texts, language_src, language_dst):
    tmp_lang_tokenizer, tmp_lang_model = download('Helsinki-NLP/opus-mt-en-ROMANCE')
    device = tmp_lang_model.cuda().device
    src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-ROMANCE-en')

    # Translate from source to target language
    translated = translate(texts, tmp_lang_model.to(device=device), tmp_lang_tokenizer, language_dst)

    # Translate from target language back to source language
    back_translated = translate(translated, src_lang_model.to(device=device), src_lang_tokenizer, language_src)

    return back_translated


if __name__ == "__main__":
    df = pd.read_csv("train.csv", header=0,
                     names=("id", "text", "is_humor", "humor_rating", "humor_controversy", "offense_rating"))
    # x = df["text"].tolist()
    data_text, label = df["text"].tolist(), df["is_humor"].tolist()

    # print(data_text)
    # back_texts = back_translate(data_text, "en", "fr")
    # print(back_texts)
    # back_texts = ["text"] + back_texts
    # label = ["is_humor"] + label
    # rows = zip(back_texts, label)
    count = 0
    with open("./translated_train.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "is_humor"])
        for d, l in zip(data_text, label):
            b = back_translate([d], "en", "fr")
            writer.writerow([b[0], l])
            count += 1
            print(count)
