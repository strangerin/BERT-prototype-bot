import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from text_processing.goemotions_resources.model_goemotions import BertForMultiLabelClassification
from text_processing.goemotions_resources.multilabel_pipeline_goemotions import MultiLabelPipeline


class GoemotionsAnalyzerBasic:
    """27 emotions + neutral"""

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

        self.goemotions = MultiLabelPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            threshold=0.3
        )


class GoEmotionsAnalyzerGroup:
    """positive, negative, ambiguous + neutral"""

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-group")
        self.model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-group")

        self.goemotions = MultiLabelPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            threshold=0.3
        )


class BertClassifier(nn.Module):
    """Custom, 11 classes, trained on internal CRM data"""

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        # second argument is num of our pre-defined classes
        self.linear = nn.Linear(768, 11)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def tokenize_data(text):
    """
    :param text: Input text, user email
    :return: tokenized text object
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenized_text = tokenizer(str(text)[:512],
                               padding='max_length', max_length=512, truncation=True,
                               return_tensors="pt")
    return tokenized_text


def classify_custom(model, text):
    """
    :param model: Our pre-loaded model from main script
    :param text: Untokenized user input
    :return: A result string, for demo purposes
    """
    # TODO maybe rewrite this into a class, if need be, will use service function for now, for testing purposes
    labels_decoded = ['activation_data', 'billing_data', 'licenses_downloads_data',
                      'luminar_x_data', 'performance_stability_data', 'presale_marketing',
                      'refunds', 'skylum_account', 'spam', 'undefined', 'workflow_questions']
    tokenized_text = tokenize_data(text)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    mask = tokenized_text['attention_mask'].to(device)
    input_id = tokenized_text['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)
    # results interpretation
    main_label = labels_decoded[output.argmax()]
    possible_labels = []
    possible_results = torch.topk(output, 3)[1]

    for result in possible_results.tolist()[0]:
        possible_labels.append(labels_decoded[result])

    ret_str = f"Predicted class is: {main_label}\nTop 3 possible classes are: {possible_labels}\n"
    return ret_str
