from setfit import AbsaModel
import contractions
import nltk
import spacy
import re

# Tải về các gói cần thiết từ NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
# Tải mô hình ngôn ngữ tiếng Anh
nlp = spacy.load("en_core_web_sm")

def processing_text(text, lower_case=True):
    # Preprocess text: lowercasing, contraction expansion, and tokenization
    lower_text = text.lower() if lower_case else text
    expanded_text = contractions.fix(lower_text)
    processed_text = re.sub(r'[^a-zA-Z0-9\s\']', ' ', expanded_text)
    tokens = nltk.word_tokenize(processed_text)
    return ' '.join(tokens)

# Create a model with a chosen sentence transformer from the Hub
model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect",
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
)

def detect_span(sentence, query_span):
    """
    Detect the position (start, end) of a query span in a given sentence.
    """
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Split the query span into tokens
    query_tokens = query_span.split()

    span_position = None
    # Find the position of the span
    for token in doc:
        # Check if the token matches the first token of the query span
        if token.text == query_tokens[0]:
            # Attempt to match the entire query span
            span = doc[token.i:token.i + len(query_tokens)]
            if span.text == query_span:
                span_position = (token.i, token.i + len(query_tokens) - 1)  # Inclusive indexing
                break
    return span_position

def predict_sentence(sentence):
    """
    Predict aspects and polarities for a given sentence and label tokens.
    """
    # Preprocess the sentence
    processed_sentence = processing_text(sentence)
    # processed_sentence = sentence
    # Get predictions from the model
    pred = model.predict([processed_sentence])[0]

    positions, polarities = [], []
    for aspect in pred:
        # Detect span positions for each aspect term
        span_position = detect_span(processed_sentence, aspect['span'])
        if span_position:
            positions.append(span_position)
            # Map polarity to a label
            if aspect['polarity'] == 'positive':
                polarities.append('POS')
            elif aspect['polarity'] == 'negative':
                polarities.append('NEG')
            elif aspect['polarity'] == 'neutral':
                polarities.append('NEU')

    prev_end = 0

    # Tokenize the processed sentence
    tokens = nltk.word_tokenize(processed_sentence)
    labels = []

    # Assign labels to tokens based on aspect spans and polarities
    for (start, end), polarity in zip(positions, polarities):
        labels.extend(['O'] * (start - prev_end))
        if start == end:
            labels.append(f'S-{polarity}')
        else:
            labels.extend([f'B-{polarity}'] + ['I-' + polarity] * (end - start - 1) + [f'E-{polarity}'])
        prev_end = end + 1

    # Add 'O' labels for tokens outside any aspect span
    labels.extend(['O'] * (len(tokens) - prev_end))

    return list(zip(tokens, labels))

def predict_paragraph(paragraph):
    """
    Predict aspects and polarities for an entire paragraph.
    """
    # Split the paragraph into sentences using spaCy
    doc = nlp(paragraph)
    sentences = [sent.text for sent in doc.sents]
    results = []
    for sentence in sentences:
        results.append(predict_sentence(sentence))

    return results