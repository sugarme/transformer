package bert

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"reflect"

	"github.com/sugarme/transformer/util"
)

// Map model name to url
var configMap map[string]string = map[string]string{
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

// BertConfig defines the BERT model architecture (i.e., number of layers,
// hidden layer size, label mapping...)
type BertConfig struct {
	HiddenAct                 string           `json:"hidden_act"`
	AttentionProbsDropoutProb float64          `json:"attention_probs_dropout_prob"`
	HiddenDropoutProb         float64          `json:"hidden_dropout_prob"`
	HiddenSize                int64            `json:"hidden_size"`
	InitializerRange          float32          `json:"initializer_range"`
	IntermediateSize          int64            `json:"intermediate_size"`
	MaxPositionEmbeddings     int64            `json:"max_position_embeddings"`
	NumAttentionHeads         int64            `json:"num_attention_heads"`
	NumHiddenLayers           int64            `json:"num_hidden_layers"`
	TypeVocabSize             int64            `json:"type_vocab_size"`
	VocabSize                 int64            `json:"vocab_size"`
	OutputAttentions          bool             `json:"output_attentions"`
	OutputHiddenStates        bool             `json:"output_hidden_states"`
	IsDecoder                 bool             `json:"is_decoder"`
	Id2Label                  map[int64]string `json:"id_2_label"`
	Label2Id                  map[string]int64 `json:"label_2_id"`
	NumLabels                 int64            `json:"num_labels"`
}

func ConfigFromFile(filename string) (*BertConfig, error) {
	filePath, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}

	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	buff, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	var config BertConfig
	err = json.Unmarshal(buff, &config)
	if err != nil {
		fmt.Println(err)
		log.Fatalf("Could not parse configuration to BertConfiguration.\n")
	}
	return &config, nil
}

// Load loads model configuration from file or model name. It also updates
// default configuration parameters if provided.
// This method implements `PretrainedConfig` interface.
func (c *BertConfig) Load(modelNameOrPath string, params map[string]interface{}) error {

	var urlOrFilename string
	// If modelName, infer to default configuration filename:
	if configFile, ok := configMap[modelNameOrPath]; ok {
		urlOrFilename = configFile
	} else {
		// Otherwise, just take the input
		urlOrFilename = modelNameOrPath
	}

	cachedFile, err := util.CachedPath(urlOrFilename)
	if err != nil {
		return err
	}

	err = c.fromFile(cachedFile)
	if err != nil {
		return err
	}

	// Update custom parameters
	c.updateParams(params)

	return nil
}

func (c *BertConfig) fromFile(filename string) error {
	filePath, err := filepath.Abs(filename)
	if err != nil {
		return err
	}

	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	buff, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}

	err = json.Unmarshal(buff, c)
	if err != nil {
		fmt.Println(err)
		log.Fatalf("Could not parse configuration to BertConfiguration.\n")
	}

	return nil
}

func (c *BertConfig) GetVocabSize() int64 {
	return c.VocabSize
}

func (c *BertConfig) updateParams(params map[string]interface{}) {
	for k, v := range params {
		c.updateField(k, v)
	}
}

func (c *BertConfig) updateField(field string, value interface{}) {
	// Check whether field name exists
	if reflect.ValueOf(c).Elem().FieldByName(field).IsValid() {
		// Check whether same type
		if reflect.ValueOf(c).Elem().FieldByName(field).Kind() == reflect.TypeOf(value).Kind() {
			reflect.ValueOf(c).Elem().FieldByName(field).Set(reflect.ValueOf(value))
		}
	}
}
