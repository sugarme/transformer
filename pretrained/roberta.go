package pretrained

// RobertaConfigs is a map of pretrained Roberta configuration names to corresponding URLs.
var RobertaConfigs map[string]string = map[string]string{
	"roberta-base":       "https://cdn.huggingface.co/roberta-base-config.json",
	"roberta-qa":         "https://s3.amazonaws.com/models.huggingface.co/bert/deepset/roberta-base-squad2/config.json",
	"xlm-roberta-ner-en": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-config.json",
	"xlm-roberta-ner-de": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-config.json",
	"xlm-roberta-ner-nl": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-config.json",
	"xlm-roberta-ner-es": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-config.json",
}

// RobertaModels is a map of pretrained Roberta model names to corresponding URLs.
var RobertaModels map[string]string = map[string]string{
	"roberta-base":       "https://cdn.huggingface.co/roberta-base-rust_model.ot",
	"roberta-qa":         "https://cdn.huggingface.co/deepset/roberta-base-squad2/rust_model.ot",
	"xlm-roberta-ner-en": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll03-english-rust_model.ot",
	"xlm-roberta-ner-de": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll03-german-rust_model.ot",
	"xlm-roberta-ner-nl": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll02-dutch-rust_model.ot",
	"xlm-roberta-ner-es": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll02-spanish-rust_model.ot",
}

// RobertaVocabs is a map of pretrained Roberta vocab name to corresponding URLs.
var RobertaVocabs map[string]string = map[string]string{
	"roberta-base":       "https://cdn.huggingface.co/roberta-base-vocab.json",
	"roberta-qa":         "https://cdn.huggingface.co/deepset/roberta-base-squad2/vocab.json",
	"xlm-roberta-ner-en": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll03-english-sentencepiece.bpe.model",
	"xlm-roberta-ner-de": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll03-german-sentencepiece.bpe.model",
	"xlm-roberta-ner-nl": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll02-dutch-sentencepiece.bpe.model",
	"xlm-roberta-ner-es": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll02-spanish-sentencepiece.bpe.model",
}

// RobertaMerges is a map of pretrained Roberta vocab merges name to corresponding URLs.
var RobertaMerges map[string]string = map[string]string{
	"roberta-base": "https://cdn.huggingface.co/roberta-base-merges.txt",
	"roberta-qa":   "https://cdn.huggingface.co/deepset/roberta-base-squad2/merges.txt",
}
