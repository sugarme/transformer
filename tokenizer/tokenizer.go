package tokenizer

type Offset struct {
	Start uint
	End   uint
}

type PreToken struct {
	Value  string
	Offset Offset
}

type Token struct {
	Id uint32
	PreToken
}

// PreTokenizer processes strings before going to the model
type PreTokenizer interface {
	PreTokenize(s string) []PreToken
}

// Model represents a model used during tokenization (i.e., BPE, Word, or Unigram)
type Model interface {
	Tokenize(tokens []PreToken) []Token
	TokenToId(token string) (uint32, error)
	IdToToken(id uint32) (string, error)
	GetVocabSize() uint
	Save(path string, name string) error
}

type PostProcess interface {
	// Returns the number of tokens that will be added during the processing step
	AddedTokens(isPair bool) uint
	// Process processes both encodings and returns a new merged one
	// NOTE: pairEncoding is optional
	Process(...Encoding) Encoding
}
