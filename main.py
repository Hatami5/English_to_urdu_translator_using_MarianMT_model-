from transformers import MarianMTModel, MarianTokenizer

# Specify the model for English to Urdu translation
model_name = 'Helsinki-NLP/opus-mt-en-ur'

# Load the model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_to_urdu(text):
    # Tokenize the input text
    tokens = tokenizer.encode(text, return_tensors='pt')
    # Generate translation
    translated = model.generate(tokens)
    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example usage
text_to_translate = input("type your text to translate: ")
urdu_translation = translate_to_urdu(text_to_translate)
print(f'Translation: {urdu_translation}')


