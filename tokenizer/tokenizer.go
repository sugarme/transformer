package tokenizer

// tokenizer represents a tokenization pipeline
// TODO: full description

import (
	"log"
	"regexp"

	"github.com/sugarme/sermo/normalizer"
)

type Offsets struct {
	Start uint
	End   uint
}

type PreToken struct {
	Value   string
	Offsets Offsets
}

type Token struct {
	Id      uint32
	Value   string
	Offsets Offsets
}

// PreTokenizer processes strings before going to the model
type PreTokenizer interface {
	PreTokenize(s string) []PreToken
}

// Model represents a model used during tokenization (i.e., BPE, Word, or Unigram)
type Model interface {
	Tokenize(tokens []PreToken) []Token
	TokenToId(token string) uint32
	IdToToken(id uint32) string
	GetVocabSize() uint
	Save(path string, name string) error
}

type PostProcessor interface {
	// Returns the number of tokens that will be added during the processing step
	AddedTokens(isPair bool) uint
	// Process processes both encodings and returns a new merged one
	// NOTE: pairEncoding is optional
	Process(...Encoding) Encoding
}

// Decoder takes care of (merges) the given slice of tokens to string
type Decoder interface {
	Decode(tokens []string) string
}

// Trainer is responsible for training a model. It takes lines/sentences
// and returns a tokenizer `Model` when done.
type Trainer interface {
	// Whether showing progress bar or not
	WithProgressBar() bool
	// Actual training method. It will return a trained model and
	// a list of `special tokens` to be added directly to the tokenizer
	// along with the model
	Train(words map[string]uint32) (Model, []string)
	// ProcessTokens processes a bunch of tokens and counts them as relevant
	ProcessTokens(words map[string]uint32, tokens []string)
}

// Implement methods for `Token`
// NewToken generate new token from input data
func NewToken(id uint32, value string, offsets Offsets) Token {
	return Token{
		Id:      id,
		Value:   value,
		Offsets: offsets,
	}
}

type Single string
type Dual struct {
	Value string
	Pair  string
}

type EncodeInputType []interface{}

var EncodeInput EncodeInputType = []interface{}{
	"Single",
	Dual{
		Value: "Value",
		Pair:  "Pair",
	},
}

// var EncodeInput map[interface{}]string = map[interface{}]string{
// "Single": "Single",
// Dual{
// Value: "Value",
// Pair:  "Pair",
// }: "Dual",
// }

type AddedToken struct {
	Content      string // content of the added token
	IsSingleWord bool   // whether this token is single word or break words
}

// AddedTokenFrom creates AddedToken from input content
func AddedTokenFrom(content string, opt ...bool) AddedToken {
	var isSingleWord bool = false
	if len(opt) > 0 {
		isSingleWord = opt[0]
	}
	return AddedToken{
		Content:      content,
		IsSingleWord: isSingleWord,
	}
}

// Default initiates an addedtoken with default value
func (at *AddedToken) Default() {
	at = &AddedToken{
		Content:      "",
		IsSingleWord: false,
	}
}

// Hash hashes on the content of addedtoken
// Should we use SipHash or use map?
func (at *AddedToken) Hash() {

}

// Eq checks whether current addedtoken content is equal to other
func (at *AddedToken) Eq(other AddedToken) bool {
	return at.Content == other.Content
}

type TruncationParams struct{}
type PaddingParams struct{}

// Tokenizer represents a tokenization pipeline.
// It can implement any encoding or decoding of any text.
type Tokenizer struct {
	// Parts
	Normalizer    normalizer.Normalizer
	PreTokenizer  PreTokenizer
	Model         Model
	PostProcessor PostProcessor
	Decoder       Decoder

	// Vocab
	AddedTokens   map[AddedToken]uint32
	AddedTokensR  map[uint32]AddedToken
	SplitRe       *regexp.Regexp
	SpecialTokens map[string]uint32

	// General processing parameters
	Trunc   TruncationParams
	Padding PaddingParams
}

// Implementing methods for Tokenizer
func NewTokenizer(model Model) Tokenizer {
	return Tokenizer{
		Normalizer:    nil,
		PreTokenizer:  nil,
		Model:         model,
		PostProcessor: nil,
		Decoder:       nil,

		AddedTokens:   make(map[AddedToken]uint32),
		AddedTokensR:  make(map[uint32]AddedToken),
		SplitRe:       &regexp.Regexp{},
		SpecialTokens: make(map[string]uint32),

		Trunc:   TruncationParams{},
		Padding: PaddingParams{},
	}
}

func (t *Tokenizer) WithNormalizer(n normalizer.Normalizer) {
	t.Normalizer = n
}

func (t *Tokenizer) GetNormalizer() normalizer.Normalizer {
	return t.Normalizer
}

func (t *Tokenizer) WithPreTokenizer(preTokenizer PreTokenizer) {
	t.PreTokenizer = preTokenizer
}

func (t *Tokenizer) GetPreTokenizer() PreTokenizer {
	return t.PreTokenizer
}

func (t *Tokenizer) WithPostProcessor(postProcessor PostProcessor) {
	t.PostProcessor = postProcessor
}

func (t *Tokenizer) GetPostProcessor() PostProcessor {
	return t.PostProcessor
}

func (t *Tokenizer) WithDecoder(decoder Decoder) {
	t.Decoder = decoder
}

func (t *Tokenizer) GetDecoder() Decoder {
	return t.Decoder
}

func (t *Tokenizer) WithModel(model Model) {
	t.Model = model
}

func (t *Tokenizer) GetModel() Model {
	return t.Model
}

func (t *Tokenizer) WithTruncation(trunc TruncationParams) {
	t.Trunc = trunc
}

func (t *Tokenizer) GetTruncation() TruncationParams {
	return t.Trunc
}

func (t *Tokenizer) WithPadding(padding PaddingParams) {
	t.Padding = padding
}

func (t *Tokenizer) GetVocabSize(withAddedToken bool) uint {
	if withAddedToken {
		return t.Model.GetVocabSize() + uint(len(t.AddedTokens))
	}

	return t.Model.GetVocabSize()
}

func (t *Tokenizer) TokenToId(token string) uint32 {

	addedToken := AddedTokenFrom(token)

	id, ok := t.AddedTokens[addedToken]
	if ok {
		return id
	}

	return t.Model.TokenToId(token)

}

func (t *Tokenizer) IdToToken(id uint32) string {
	tok, ok := t.AddedTokensR[id]
	if ok {
		return tok.Content
	}
	return t.Model.IdToToken(id)
}

func (t *Tokenizer) NumAddedTokens(isPair bool) uint {
	return t.PostProcessor.AddedTokens(isPair)
}

type splitRes struct {
	Content string
	Id      uint32
	Found   bool // whether split was found in AddedTokens/SpecialTokens
}

// Encode encodes the given sentence
func (t *Tokenizer) Encode(input EncodeInputType) Encoding {
	generateOutput := func(sentence string, typeId uint32) Encoding {
		// Split into as many sequences as needed to avoid splitting
		// on our added tokens
		var splits []splitRes
		var encodings []Encoding

		splits = t.splitOnAddedTokens(sentence)
		for _, s := range splits {
			// If this is one of our added tokens, return an encoding directly
			if s.Found {
				e := NewEncoding(*normalizer.NewNormalizedFrom(s.Content), []uint32{s.Id}, []uint32{typeId}, []string{s.Content}, []Offsets{{0, uint(len(s.Content))}}, []uint32{0}, []uint32{1}, []Encoding{})

				encodings = append(encodings, e)
			}

			// 1. Normalization
			var normalized normalizer.Normalized
			if (normalizer.NewNormalizer()) != t.Normalizer { // make sure that normalizer is included
				normalized, err := t.Normalizer.Normalize(s.Content)
				if err != nil {
					log.Fatal(err)
				}
			}
			// 2. Pre-tokenization
			var preTokenized []PreToken

			// TODO: check whether preTokenizer is included
			preTokenized = t.PreTokenizer.PreTokenize(normalized.Get().Normalized)

			// 3. Model
			// TODO: check whether model is included
			output := t.Model.Tokenize(preTokenized)

			var en Encoding

			for _, t := range output {
				en.Ids = append(en.Ids, t.Id)
				en.Tokens = append(en.Tokens, t.Value)
				en.Offsets = append(en.Offsets, t.Offsets)
			}

			for i := range output {
				en.TypeIds[i] = typeId
				en.SpecialTokenMask[i] = 0
				en.AttentionMask[i] = 1
			}

			en.Overflowing = []Encoding{}

			encodings = append(encodings, en)
		} // end loop over splits

		if len(encodings) == 0 {
			// TODO: create a new default Encoding
			return Encoding.Default()
		}

	}

	return nil

}

func (t *Tokenizer) splitOnAddedTokens(sentence string) []splitRes {

	var splits []splitRes
	rs := []rune(sentence)
	var allSplits [][]int

	// if there's no splitRe (regular epxression to split), do nothing
	if t.SplitRe == nil {
		splits = append(splits, splitRes{sentence, 0, false})
		return splits
	}

	// matches contains slice of 2-element items (start and end byte position)
	// of the matched strings
	matches := t.SplitRe.FindAllStringIndex(sentence, -1)

	for _, m := range matches {
		splits = append(splits, splitRes{
			Content: string(rs[m[0]:m[1]]),
			Id:      0,
		})
	}

	// Collect also the splits in-between added tokens
	startOffset := 0
	for _, m := range matches {
		if startOffset < m[0] {
			allSplits = append(allSplits, []int{startOffset, m[0]})
		}

		allSplits = append(allSplits, []int{m[0], m[1]})
		startOffset = m[1]
	}

	// Check for the last piece
	last := allSplits[len(allSplits)]
	if last[1] < len(sentence) {
		allSplits = append(allSplits, []int{last[1], len(sentence)})
	}

	if len(allSplits) == 0 {
		splits = append(splits, splitRes{sentence, 0, false})
		return splits
	}

	for _, m := range allSplits {
		s := string(rs[m[0]:m[1]])
		// Look up at special tokens
		id, ok := t.SpecialTokens[s]
		// not found. Look up at added tokens
		if !ok {
			// If not found, id will be 0 and ok = false
			id, ok = t.AddedTokens[AddedToken{
				Content:      s,
				IsSingleWord: false,
			}]
			if !ok {
				splits = append(splits, splitRes{
					Content: s,
					Id:      0,
					Found:   false,
				})
			}
		}
		splits = append(splits, splitRes{
			Content: s,
			Id:      id,
			Found:   true,
		})
	}

	return splits

}
