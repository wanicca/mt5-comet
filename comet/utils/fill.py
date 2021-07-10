from transformers import pipeline

nlp = pipeline("fill-mask")
print(nlp(f"Event: PersonX throws {nlp.tokenizer.mask_token} on the subject."))
print(nlp(f"PersonX goes and bought {nlp.tokenizer.mask_token}."))
