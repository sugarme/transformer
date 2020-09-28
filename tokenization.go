package transformer

import (
	"fmt"
	"log"

	"github.com/sugarme/tokenizer"
)

// Tokenizer is alias of corresponding struct in `github.com/sugarme/tokenizer` package.
type Tokenizer = tokenizer.Tokenizer

type PretrainedTokenizer interface {
	TokenizerFromPretrained(string) *Tokenizer
}

var (
	TokenizerMapping     map[string][]string = make(map[string][]string)
	SlowTokenizerMapping map[string]string   = make(map[string]string)
)

func init() {
	TokenizerMapping = map[string][]string{
		"Bert": []string{"BertTokenizer", "BertTokenizerFast"},

		// TODO: appending
	}

	for k, v := range TokenizerMapping {
		if len(v) > 0 {
			SlowTokenizerMapping[k] = v[0]
		}
	}
}

// TokenizerFromPretrained initiate Tokenizer from pretrained local or remote file.
// It will load from cached file if existing, otherwise, download file then caching
// for next time loading.
func TokenizerFromPretrained(pretrainedModelNameOrPath string, customParams map[string]interface{}) *Tokenizer {
	useFast := false
	if v, ok := customParams["useFast"]; ok {
		useFast = v.(bool)
	}

	config := ConfigFromPretrained(pretrainedModelNameOrPath, customParams)
	modelType := config.ModelType

	toks, ok := TokenizerMapping[modelType]
	if !ok {
		var modelTypes string
		for k, _ := range TokenizerMapping {
			modelTypes += fmt.Sprintf("%v, ", k)
		}
		log.Fatalf("Unrecognized configuration class '%v' to build Tokenizer.\nModel type should be one of %v.\n", modelType, modelTypes)
	}

	if useFast && len(toks) > 1 {
		customParams["tokenizerClass"] = toks[1]
	} else {
		customParams["tokenizerClass"] = toks[0]
	}

	return tokenizerFromPretrained(pretrainedModelNameOrPath, customParams)
}

func tokenizerFromPretrained(pretrainedModelNameOrPath string, params map[string]interface{}) *Tokenizer {

	// TODO: implement

	panic("Function not implemented yet.")
}
