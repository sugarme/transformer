package pretrained

// BertConfigs is a map of pretrained Bert configuration names to corresponding URLs
var BertConfigs map[string]string = map[string]string{
	"bert-base-uncased":                                     "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
	"bert-large-uncased":                                    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
	"bert-base-cased":                                       "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
	"bert-large-cased":                                      "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
	"bert-base-multilingual-uncased":                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
	"bert-base-multilingual-cased":                          "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
	"bert-base-chinese":                                     "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
	"bert-base-german-cased":                                "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
	"bert-large-uncased-whole-word-masking":                 "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
	"bert-large-cased-whole-word-masking":                   "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
	"bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
	"bert-large-cased-whole-word-masking-finetuned-squad":   "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
	"bert-base-cased-finetuned-mrpc":                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
	"bert-base-german-dbmdz-cased":                          "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json",
	"bert-base-german-dbmdz-uncased":                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-config.json",
	"cl-tohoku/bert-base-japanese":                          "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese/config.json",
	"cl-tohoku/bert-base-japanese-whole-word-masking":       "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking/config.json",
	"cl-tohoku/bert-base-japanese-char":                     "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char/config.json",
	"cl-tohoku/bert-base-japanese-char-whole-word-masking":  "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking/config.json",
	"TurkuNLP/bert-base-finnish-cased-v1":                   "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/config.json",
	"TurkuNLP/bert-base-finnish-uncased-v1":                 "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/config.json",
	"wietsedv/bert-base-dutch-cased":                        "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/config.json",
}

// BertModels is a map of pretrained Bert model names to corresponding URLs
var BertModels map[string]string = map[string]string{
	"bert-base-uncased": "https://cdn.huggingface.co/bert-base-uncased-rust_model.ot",
	// TODO: update
}

var BertVocabs map[string]string = map[string]string{
	"bert-base-uncased":                                     "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
	"bert-large-uncased":                                    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
	"bert-base-cased":                                       "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
	"bert-large-cased":                                      "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
	"bert-base-multilingual-uncased":                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
	"bert-base-multilingual-cased":                          "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
	"bert-base-chinese":                                     "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
	"bert-base-german-cased":                                "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
	"bert-large-uncased-whole-word-masking":                 "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
	"bert-large-cased-whole-word-masking":                   "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
	"bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
	"bert-large-cased-whole-word-masking-finetuned-squad":   "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
	"bert-base-cased-finetuned-mrpc":                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
	"bert-base-german-dbmdz-cased":                          "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
	"bert-base-german-dbmdz-uncased":                        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
	"TurkuNLP/bert-base-finnish-cased-v1":                   "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/vocab.txt",
	"TurkuNLP/bert-base-finnish-uncased-v1":                 "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/vocab.txt",
	"wietsedv/bert-base-dutch-cased":                        "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/vocab.txt",
}
