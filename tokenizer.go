package transformer

import (
	"github.com/sugarme/transformer/pretrained"
)

// LoadTokenizer loads pretrained tokenizer from local or remote file.
//
// Parameters:
// - `tk` pretrained.Tokenizer (any tokenizer model that implements pretrained `Tokenizer` interface)
// - `modelNameOrPath` is a string of either
//		+ Model name or
// 		+ File name or path or
// 		+ URL to remote file
// If `modelNameOrPath` is resolved, function will cache data using `TransformerCache`
// environment if existing, otherwise it will be cached in `$HOME/.cache/transformers/` directory.
// If `modleNameOrPath` is valid URL, file will be downloaded and cached.
// Finally, vocab data will be loaded to `tk`.
func LoadTokenizer(tk pretrained.Tokenizer, vocabNameOrPath, mergesNameOrPath string, customParams map[string]interface{}) error {
	return tk.Load(vocabNameOrPath, mergesNameOrPath, customParams)
}
