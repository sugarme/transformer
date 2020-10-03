package pretrained

// BertConfigs is a map of pretrained Bert configuration names to corresponding URLs.
var BertConfigs map[string]string = map[string]string{
	"bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
	"bert-ner":          "https://cdn.huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/config.json",
	"bert-qa":           "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",

	// Roberta
	"roberta-base":       "https://cdn.huggingface.co/roberta-base-config.json",
	"roberta-qa":         "https://s3.amazonaws.com/models.huggingface.co/bert/deepset/roberta-base-squad2/config.json",
	"xlm-roberta-ner-en": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-config.json",
	"xlm-roberta-ner-de": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-config.json",
	"xlm-roberta-ner-nl": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-config.json",
	"xlm-roberta-ner-es": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-config.json",
}

// BertModels is a map of pretrained Bert model names to corresponding URLs.
var BertModels map[string]string = map[string]string{
	"bert-base-uncased": "https://cdn.huggingface.co/bert-base-uncased-rust_model.ot",
	"bert-ner":          "https://cdn.huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/rust_model.ot",
	"bert-qa":           "https://cdn.huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad-rust_model.ot",
}

// BertVocabs is a map of BERT model vocab name to corresponding URLs.
var BertVocabs map[string]string = map[string]string{
	"bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
	"bert-ner":          "https://cdn.huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/vocab.txt",
	"bert-qa":           "https://cdn.huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
}
