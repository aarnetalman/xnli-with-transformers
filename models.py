from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
)


def get_model(config):
    if config.model == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=3
        )
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-large", num_labels=3
        )
    return tokenizer, model
