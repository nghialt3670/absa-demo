import nltk
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from safetensors.torch import load_file
# from transformers import BertConfig, BertTokenizer

# from inference import predict
# from models import BertABSATagger
from pretrained_inference import predict_paragraph

# nltk.download("punkt_tab")

# tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)

# config_path = r"../checkpoints/1600/config.json"
# config = BertConfig.from_json_file(config_path)

# model_path = r"../checkpoints/1600/model.safetensors"
# bert_tagger = BertABSATagger(config)

# state_dict = load_file(model_path)
# bert_tagger.load_state_dict(state_dict)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/api/tag")
# async def tag(text: str = Body(...)):
#     tags = predict(text, tokenizer, bert_tagger, "cpu")
#     return tags

@app.post("/api/tag")
async def tag(text: str = Body(...)):
    tags = predict_paragraph(text)
    return tags