import torch
from transformers import BertConfig, BertModel, PreTrainedModel, load_tf_weights_in_bert

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}


class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 1024
        self.n_rnn_layers = 1  # not used if tagger is non-RNN model
        self.bidirectional = True  # not used if tagger is non-RNN model


class BertLayerNorm(torch.nn.Module):
    """
    A BERT layer normalization module.

    Args:
        hidden_size (int): The size of the hidden layers.
        eps (float): The epsilon value to prevent division by zero during normalization.
    """

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # Compute mean and variance for layer normalization
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        normalized_x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * normalized_x + self.bias


class BertPreTrainedModel(PreTrainedModel):
    """
    Abstract class to handle weights initialization and loading pretrained BERT models.

    Args:
        config (BertConfig): Configuration class for BERT model.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert  # Method to load TF weights in BERT
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """
        Initializes weights for various modules in BERT model.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Initialize weights with a normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            # Initialize LayerNorm weights and biases
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            # Initialize Linear layer biases
            module.bias.data.zero_()


class SAN(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Apply self-attention and residual cotorch.nnection."""
        # Apply multi-head attention
        src2, _ = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout(src2)
        # Apply layer normalization
        return self.norm(src)


class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BertABSATagger, self).__init__(bert_config)

        self.tagger_config = TaggerConfig()
        self.tagger_config.absa_type = bert_config.absa_type

        self.num_labels = bert_config.num_labels
        self.bert = self._get_bert_model(bert_config)
        if bert_config.fix_tfm:
            self._freeze_bert_parameters()

        self.bert_dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        self.tagger, penultimate_hidden_size = self._init_tagger(bert_config)
        self.classifier = torch.nn.Linear(penultimate_hidden_size, self.num_labels)

    def _get_bert_model(self, bert_config):
        if bert_config.tfm_mode == "finetune":
            return BertModel(bert_config)
        return self._raise_exception(
            f"Invalid transformer mode {bert_config.tfm_mode}!!!"
        )

    def _freeze_bert_parameters(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def _init_tagger(self, bert_config):
        tagger_type = self.tagger_config.absa_type

        if tagger_type == "linear":
            tagger, hidden_size = (None, bert_config.hidden_size)
        elif tagger_type == "tfm":
            tagger, hidden_size = (
                torch.nn.TransformerEncoderLayer(
                    d_model=bert_config.hidden_size,
                    nhead=16,
                    dim_feedforward=4 * bert_config.hidden_size,
                    dropout=0.1,
                ),
                bert_config.hidden_size,
            )
        elif tagger_type == "san":
            tagger, hidden_size = (
                SAN(d_model=bert_config.hidden_size, nhead=16, dropout=0.1),
                bert_config.hidden_size,
            )
        else:
            self._raise_exception(f"Unimplemented downstream tagger {tagger_type}...")

        return tagger, hidden_size

    @staticmethod
    def _raise_exception(message):
        raise Exception(message)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        tagger_input = self.bert_dropout(outputs[0])
        logits = (
            self._process_tagger_input(tagger_input)
            if self.tagger
            else self.classifier(tagger_input)
        )
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss = self._compute_loss(logits, labels, attention_mask)
            outputs = (loss,) + outputs
        return outputs

    def _process_tagger_input(self, tagger_input):
        classifier_input = self.tagger(tagger_input.transpose(0, 1)).transpose(0, 1)
        return self.classifier(self.bert_dropout(classifier_input))

    def _compute_loss(self, logits, labels, attention_mask):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            return loss_fct(active_logits, active_labels)
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
