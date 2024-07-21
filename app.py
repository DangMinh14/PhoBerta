import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
from transformers.models.roberta.modeling_roberta import *
from transformers import RobertaForQuestionAnswering
from nltk import word_tokenize
import json
import pandas as pd
import re
import base64
# Set the background image
# background_image = """
# <style>
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
#     background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
#     background-position: center;  
#     background-repeat: no-repeat;
# }
# </style>
# """

# st.markdown(background_image, unsafe_allow_html=True)

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack("Background.png")

image_url = "logo1.png"

# Hiển thị hình ảnh mà không có caption và điều chỉnh kích thước nhỏ lại
st.image(image_url, width=100)

# Download punkt for nltk
nltk.download('punkt')

# Load PhoBert model and tokenizer
phoBert_model = AutoModelForSequenceClassification.from_pretrained('minhdang14902/PhoBert_Edu')
phoBert_tokenizer = AutoTokenizer.from_pretrained('minhdang14902/PhoBert_Edu')
chatbot_pipeline = pipeline("sentiment-analysis", model=phoBert_model, tokenizer=phoBert_tokenizer)

# Load spaCy Vietnamese model
# nlp = spacy.load('vi_core_news_lg')

# Load intents from json file
def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

filename = './QA_Legal_converted_merged.json'
intents = load_json_file(filename)

def create_df():
    df = pd.DataFrame({
        'Pattern': [],
        'Tag': []
    })
    return df

df = create_df()

def extract_json_info(json_file, df):
    for intent in json_file['intents']:
        for pattern in intent['patterns']:
            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag
    return df

df = extract_json_info(intents, df)
df2 = df.copy()

labels = df2['Tag'].unique().tolist()
labels = [s.strip() for s in labels]
num_labels = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

# def tokenize_with_spacy(text):
#     doc = nlp(text)
#     tokens = [token.text for token in doc]
#     tokenized_text = ' '.join(tokens)
#     tokenized_text = re.sub(r'(?<!\s)([.,?])', r' \1', tokenized_text)
#     tokenized_text = re.sub(r'([.,?])(?!\s)', r'\1 ', tokenized_text)
#     return tokenized_text

# Load Roberta model and tokenizer

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"


class MRCQuestionAnswering(RobertaPreTrainedModel):
    config_class = RobertaConfig

    def _reorder_cache(self, past, beam_idx):
        pass

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            words_lengths=None,
            start_idx=None,
            end_idx=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            span_answer_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,  # Roberta doesn't use token_type_ids
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        context_embedding = sequence_output

        batch_size = input_ids.shape[0]
        max_sub_word = input_ids.shape[1]
        max_word = words_lengths.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))

        for i, sample_length in enumerate(words_lengths):
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0

        align_matrix = align_matrix.to(context_embedding.device)
        context_embedding_align = torch.bmm(align_matrix, context_embedding)

        logits = self.qa_outputs(context_embedding_align)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

roberta_model_checkpoint = "minhdang14902/Roberta_edu"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_checkpoint)
roberta_model = MRCQuestionAnswering.from_pretrained(roberta_model_checkpoint)

def chatRoberta(text):
    label = label2id[chatbot_pipeline(text)[0]['label']]
    response = intents['intents'][label]['responses']

    QA_input = {
        'question': text,
        'context': response[0]
    }

    # Tokenize input
    encoded_input = tokenize_function(QA_input, roberta_tokenizer)

    # Prepare batch samples
    batch_samples = data_collator([encoded_input], roberta_tokenizer)

    # Model prediction
    roberta_model.eval()
    with torch.no_grad():
        inputs = {
            'input_ids': batch_samples['input_ids'],
            'attention_mask': batch_samples['attention_mask'],
            'words_lengths': batch_samples['words_lengths'],
        }
        outputs = roberta_model(**inputs)

    # Extract answer
    result = extract_answer([encoded_input], outputs, roberta_tokenizer)
    return result

def tokenize_function(example, tokenizer):
    question_word = word_tokenize(example["question"])
    context_word = word_tokenize(example["context"])

    question_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in question_word]
    context_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in context_word]
    valid = True
    if len([j for i in question_sub_words_ids + context_sub_words_ids for j in i]) > tokenizer.model_max_length - 1:
        valid = False

    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > tokenizer.model_max_length:
        valid = False

    words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]

    return {
        "input_ids": input_ids,
        "words_lengths": words_lengths,
        "valid": valid
    }

def data_collator(samples, tokenizer):
    if len(samples) == 0:
        return {}

    def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    input_ids = collate_tokens([torch.tensor(item['input_ids']) for item in samples], pad_idx=tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(samples)):
        attention_mask[i][:len(samples[i]['input_ids'])] = 1
    words_lengths = collate_tokens([torch.tensor(item['words_lengths']) for item in samples], pad_idx=0)

    batch_samples = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'words_lengths': words_lengths,
    }

    return batch_samples

def extract_answer(inputs, outputs, tokenizer):
    plain_result = []
    for sample_input, start_logit, end_logit in zip(inputs, outputs.start_logits, outputs.end_logits):
        sample_words_length = sample_input['words_lengths']
        input_ids = sample_input['input_ids']
        answer_start = sum(sample_words_length[:torch.argmax(start_logit)])
        answer_end = sum(sample_words_length[:torch.argmax(end_logit) + 1])

        if answer_start <= answer_end:
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            if answer == tokenizer.bos_token:
                answer = ''
        else:
            answer = ''

        score_start = torch.max(torch.softmax(start_logit, dim=-1)).cpu().detach().numpy().tolist()
        score_end = torch.max(torch.softmax(end_logit, dim=-1)).cpu().detach().numpy().tolist()
        plain_result.append({
            "answer": answer,
            "score_start": score_start,
            "score_end": score_end
        })
    return plain_result

st.title("Chatbot Interface")
st.write("Hi! I am your virtual assistant. Feel free to ask, and I'll do my best to provide you with answers and assistance.")
text = st.text_input("User: ")

if st.button("Submit"):
    if text:
        result = chatRoberta(text)
        st.write(f"Chatbot: {result[0]['answer']}")
    else:
        st.write("Please enter a message.")