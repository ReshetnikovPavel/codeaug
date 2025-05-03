from torch import nn
from transformers import RobertaModel


MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MODEL_NAME = "microsoft/codebert-base"


class CodeBERTComparator(nn.Module):
    def __init__(self):
        super(CodeBERTComparator, self).__init__()
        self.codebert = RobertaModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.codebert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
