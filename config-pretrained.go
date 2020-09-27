package transformer

import (
	"reflect"
)

// BERTPretrainedConfigs is a map of pretrained BERT configurations
// See all BERT models at https://huggingface.co/models?filter=bert
var BERTPretrainedConfigs map[string]string = map[string]string{
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

var AllPretrainedConfigs map[string]string = make(map[string]string)

var ConfigMapping map[string]reflect.Type = map[string]reflect.Type{
	// "retribert": RetriBertConfig,
	// "t5":        T5Config,
	// "bert": reflect.TypeOf(bert.BertConfig),
}

var ModelNamesMapping map[string]string = map[string]string{
	"retribert": "RetriBERT",
	"t5":        "T5",
	// TODO: update this
	"bert": "BERT",
}

func appendConfigs(configs, toAppend map[string]string) map[string]string {
	for k, v := range toAppend {
		configs[k] = v
	}

	return configs
}

func init() {
	// TODO: update this map
	AllPretrainedConfigs = appendConfigs(AllPretrainedConfigs, BERTPretrainedConfigs)
}
