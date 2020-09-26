package bert

import (
	"fmt"
	"github.com/sugarme/transformer"
)

// BertPretrainedConfigArchiveMap is a map of BERT configurations
// See all BERT models at https://huggingface.co/models?filter=bert
var BertPretrainedConfigArchiveMap map[string]string = map[string]string{
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

// NewBertConfig initiates BertConfig with given input parameters or default values.
func NewBertConfig(customParams map[string]interface{}) *transformer.Config {
	defaultValues := map[string]interface{}{
		"vocabSize":                30522,
		"hiddenSize":               768,
		"numHiddenLayers":          12,
		"numAttentionHeads":        12,
		"intermediateSize":         3072,
		"hiddenAct":                "gelu",
		"hiddenDropoutProb":        0.1,
		"attentionProbDropoutProb": 0.1,
		"maxPositionEmbeddings":    512,
		"typeVocabSize":            2,
		"initializerRange":         0.02,
		"layerNormEps":             1e-12,
		"padTokenId":               0,
		"gradientCheckpointing":    false,
	}

	bertParams := defaultValues

	if customParams != nil {
		fmt.Printf("customParams: %+v\n", customParams)
		// bertParams = transformer.AddParams(customParams, defaultValues)
		bertParams = transformer.AddParams(customParams, defaultValues)
	}

	config := transformer.NewConfig(bertParams)
	config.ModelType = "bert"

	return config
}
