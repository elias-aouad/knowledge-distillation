import torch.nn as nn
from transformers import BertModel


class BertForSequenceClassification(nn.Module):
    def __init__(self, bert, num_labels: int):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert
        in_features_classifier = self.bert.config.hidden_size
        self.classifier = nn.Linear(
            in_features=in_features_classifier, out_features=num_labels, bias=True
        )

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.bert(input_ids, attention_mask=attention_mask)[
            "last_hidden_state"
        ]
        first_token_tensor = hidden_states[:, 0]
        result = self.classifier(first_token_tensor)
        return result


def get_bert_sequence_classifier(num_labels, model_name="bert-base-uncased"):
    bert = BertModel.from_pretrained(model_name)
    model = BertForSequenceClassification(bert, num_labels=num_labels)
    return model
