package pretrained

// Tokenizer is a tokenizer model interface
type Tokenizer interface {
	Load(vocabNamOrPath, mergesNameOrPath string, params map[string]interface{}) error
}
