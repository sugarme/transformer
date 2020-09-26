package transformer

import (
	"fmt"
	"reflect"
)

// PretrainedConfig is a base for all configurations.
// It contains common parameters with default value for all model's configurations
// as well as provides methods for loading, downloading and saving configurations.
type Config struct {
	ModelType string
	Params    map[string]interface{}
}

// NewConfig creates Config from input config.
// Default values will be assigned if config Params are not provided.
func NewConfig(config map[string]interface{}) *Config {

	var configMap map[string]interface{} = make(map[string]interface{})
	defaultValues := map[string]interface{}{
		"returnDict":         false,
		"outputHiddenStates": false,
		"outputAttentions":   false,
		"useCache":           true, // Not used by all models
		"torchScript":        false,
		"useBfloat16":        false,
		"pruneHeads":         make(map[int][]int), // Dict[int, List[int]]
		"tieWordEmbeddings":  true,                // whether input and output embeddings should be tied for all MLM, LM and Seq2Seq models

		// Is decoder is used in encoder-decoder models to differentiate encoder from decoder
		"isEncoderDecoder":  false,
		"isDecoder":         false,
		"addCrossAttention": false,
		"tieEncoderDecoder": false,

		// Parameters for sequence generation
		"maxLength":            20,
		"minLength":            0,
		"doSample":             false,
		"earlyStopping":        false,
		"numBeams":             1,
		"temperature":          1.0,
		"topK":                 50,
		"topP":                 1.0,
		"repetitionPenalty":    1.0,
		"lengthPenalty":        1.0,
		"noRepeatNgramSize":    0,
		"badWordIds":           nil,
		"numReturnSequences":   1,
		"chunkSizeFeedForward": 0,

		// Fine-tuning task parameters
		"architectures":  nil,
		"finetuningTask": nil,
		"id2Label":       nil,
		"label2Id":       nil,
		"numLabels":      2,

		// Tokenizer parameters
		"tokenizerClass":      nil,
		"prefix":              nil,
		"bosTokenId":          nil,
		"padTokenId":          nil,
		"eosTokenId":          nil,
		"decoderStartTokenId": nil,

		// Task specific parameters
		"taskSpecificParams": nil,

		// TPU parameters
		"xlaDevice": nil,
	}

	configMap = AddParams(config, defaultValues)

	if configMap["id2Label"] == nil {
		configMap["numLabels"] = 2
	}

	return &Config{
		ModelType: "",
		Params:    configMap,
	}
}

// AddParams initiates parameters with default values. If an input parameter is
// specified, it will update default value with given parameter.
func AddParams(customParams map[string]interface{}, defaultValues map[string]interface{}) map[string]interface{} {

	output := defaultValues

	// Update default values with custom values
	for k, v := range customParams {
		if _, ok := output[k]; ok {
			output[k] = v
		}
	}

	// Additional parameters without default values
	for k, v := range customParams {
		if _, ok := output[k]; !ok {
			output[k] = v
		}
	}

	return output
}

// UserReturnDict returns whether or not transformer will return ModelOutput instead of tuples.
func (c *Config) UseReturnDict() bool {
	// If torchscript is set, force `returnDict`=false to avoid jit errors.
	return c.Params["returnDict"].(bool) && !c.Params["torchScript"].(bool)
}

// NumLabels returns the number of labels for classification models.
func (c *Config) NumLabels() int {
	return len(c.Params["id2Label"].(map[int]string))
}

// SetNumLabels set value for parameter `numLabels` and corresponding parameters
// `id2Label` and `label2Id`
func (c *Config) SetNumLabels(numLabels int) {
	id2Label := make(map[int]string)
	label2Id := make(map[string]int)
	for i := 0; i < numLabels; i++ {
		label := fmt.Sprintf("LABEL_%v", i)
		id2Label[i] = label
		label2Id[label] = i
	}

	c.Params["id2Label"] = id2Label
	c.Params["label2Id"] = label2Id
	c.Params["numLabels"] = numLabels
}

// TODO: implement these functions:
// ================================

func ConfigFromPretrained(pretrainedModelNameOrPath string, customParams map[string]interface{}) *Config {

	// TODO: implement this
	return nil
}

func GetConfigDict(pretrainedModelNameOrPath string, customParams map[string]interface{}) (retVal1, retVal2 map[string]interface{}) {

	// TODO: implement this
	return
}

func ConfigFromDict(config map[string]interface{}, customParams map[string]interface{}) *Config {

	// TODO: implement this
	return nil
}

func ConfigFromJSON(file string) *Config {

	// TODO: implement this
	return nil
}

// TODO: implement these methods:
// ==============================

func (c *Config) SavePretrained(path string) {
	// TODO: implement
}

func (c *Config) ToDiffDict() map[string]interface{} {

	// TODO: implement this
	return nil
}

func (c *Config) ToDict() map[string]interface{} {

	return c.Params
}

func (c *Config) ToJSON() string {

	// TODO: implement this
	return ""
}

func (c *Config) ToJSONFile(file string) {
	// TODO: implement this
}

func (c *Config) Update(config map[string]interface{}) {

	// TODO: implement this
}

var AllPretrainedConfigArchiveMap map[string]interface{} = make(map[string]interface{})

// TODO: update this map

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
