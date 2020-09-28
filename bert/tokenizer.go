package bert

import (
	"github.com/sugarme/tokenizer"
)

type BertTokenizer = tokenizer.Tokenizer
type BertTokenizerFast = tokenizer.Tokenizer

// BertJapaneseTokenizerFromPretrained initiate BERT tokenizer for Japanese language from pretrained file.
func BertJapaneseTokenizerFromPretrained(pretrainedModelNameOrPath string, customParams map[string]interface{}) *tokenizer.Tokenizer {

	// TODO: implement it

	panic("Not implemented yet.")
}
