package bert

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"reflect"
)

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

// NewBertConfig initiates BertConfig with given input parameters or default values.
func NewConfig(customParams map[string]interface{}) *BertConfig {
	defaultValues := map[string]interface{}{
		"VocabSize":                int64(30522),
		"HiddenSize":               int64(768),
		"NumHiddenLayers":          int64(12),
		"NumAttentionHeads":        int64(12),
		"IntermediateSize":         int64(3072),
		"HiddenAct":                "gelu",
		"HiddenDropoutProb":        float64(0.1),
		"AttentionProbDropoutProb": float64(0.1),
		"MaxPositionEmbeddings":    int64(512),
		"TypeVocabSize":            int64(2),
		"InitializerRange":         float32(0.02),
		"LayerNormEps":             1e-12, // not applied yet
		"PadTokenId":               0,     // not applied yet
		"GradientCheckpointing":    false, // not applied yet
	}

	params := defaultValues
	for k, v := range customParams {
		if _, ok := params[k]; ok {
			params[k] = v
		}
	}

	config := new(BertConfig)
	config.updateParams(params)

	return config
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
// This method implements `pretrained.Config` interface.
func (c *BertConfig) Load(modelNameOrPath string, params map[string]interface{}) error {
	err := c.fromFile(modelNameOrPath)
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
