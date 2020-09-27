package bert

import (
	"fmt"
	"github.com/sugarme/transformer"
)

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
